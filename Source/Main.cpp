#include <stdio.h>
#include <stdint.h>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

// Dataset include
#include "mnist_dataset.h"

// AI Models includes
#include "model_nn.h"

/*
	MNIST Dataset-based figure reading AI.
	The AI is written entirely using no libraries other than default Windows & C / C++ libraries and CUDA.
*/

#define PATH_TRAINING_SET_LABELS "D:\\Datasets\\MNIST\\train-labels.idx1-ubyte"
#define PATH_TRAINING_SET_IMAGES "D:\\Datasets\\MNIST\\train-images.idx3-ubyte"

#define PATH_TEST_SET_LABELS "D:\\Datasets\\MNIST\\t10k-labels.idx1-ubyte"
#define PATH_TEST_SET_IMAGES "D:\\Datasets\\MNIST\\t10k-images.idx3-ubyte"

#define KILOBYTES(n) (1024 * n)

MNIST_Dataset LoadMNISTDataset(const char* LabelsFile, const char* ImagesFile)
{
	uint8_t* labelsBuffer = (uint8_t*)malloc(KILOBYTES(60));
	uint8_t* imagesBuffer = (uint8_t*)malloc(KILOBYTES(50000));

	if (labelsBuffer == nullptr || imagesBuffer == nullptr)
	{
		printf("Error: Out of memory for Dataset file reading buffers.\n");
		return {};
	}

	// TRAINING LABELS
	{
		HANDLE file_trainingLabels = CreateFile(LabelsFile, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
		if (file_trainingLabels == INVALID_HANDLE_VALUE)
		{
			printf("Error opening Dataset Labels file '%s'. Error Code = %d\n", LabelsFile, GetLastError());
			return {};
		}


		DWORD readBytes = 0;
		if (!ReadFile(file_trainingLabels, labelsBuffer, KILOBYTES(60), &readBytes, NULL))
		{
			printf("Error reading Dataset Labels file '%s'. Error Code = %d\n", LabelsFile, GetLastError());
			CloseHandle(file_trainingLabels);
			return {};
		}

		CloseHandle(file_trainingLabels);
	}

	// TRAINING IMAGES
	{
		HANDLE file_trainingImages = CreateFile(ImagesFile, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
		if (file_trainingImages == INVALID_HANDLE_VALUE)
		{
			printf("Error opening Dataset Images file '%s'. Error Code = %d\n", ImagesFile, GetLastError());
			return {};
		}


		DWORD readBytes = 0;
		if (!ReadFile(file_trainingImages, imagesBuffer, KILOBYTES(46000), &readBytes, NULL))
		{
			printf("Error reading Dataset Images file '%s'. Error Code = %d\n", ImagesFile, GetLastError());
			CloseHandle(file_trainingImages);
			return {};
		}

		CloseHandle(file_trainingImages);
	}

	MNIST_Dataset Dataset = ParseDataset(labelsBuffer, imagesBuffer);

	// Verify integrity of read data
	if (Dataset.imageCount == 0 || Dataset.imageCount != Dataset.labelCount)
	{
		printf("Error: Dataset Image Count does not match Label Count (%d != %d), or failed to load appropriately !\n", Dataset.imageCount, Dataset.labelCount);
		return {};
	}

	return Dataset;
}

MNIST_Dataset LoadTrainingSet()
{
	return LoadMNISTDataset(PATH_TRAINING_SET_LABELS, PATH_TRAINING_SET_IMAGES);
}

MNIST_Dataset LoadTestSet()
{
	return LoadMNISTDataset(PATH_TEST_SET_LABELS, PATH_TEST_SET_IMAGES);
}

int Main_TestExport(int argc, char** argv)
{
	printf("Running Test Export program...\n");

	MNIST_Dataset TrainingData = LoadTrainingSet();

	if (TrainingData.imageCount == 0)
	{
		printf("Failed to load training dataset.\n");
		return 1;
	}

	printf("Training Set: Total Labeled Images Count = %d.\n", TrainingData.imageCount);

	printf("First image in dataset (Label = %d) =\n\t[\t", TrainingData.labels[0]);
	for (int y = 0; y < MNIST_Img::IMG_HEIGHT; y++)
	{
		if (y > 0)
		{
			// New Line
			printf("\n\t\t");
		}
		for (int x = 0; x < MNIST_Img::IMG_WIDTH; x++)
		{
			printf("%d\t", TrainingData.images[0].pixelValues[y * MNIST_Img::IMG_WIDTH * x]);
		}
	}
	printf("\t]\n");

	static const int EXPORT_IMG_COUNT = 5;
	printf("Exporting %d random training images as bitmaps.\n", EXPORT_IMG_COUNT);

	srand(GetTickCount64());

	for (int exportIndex = 0; exportIndex < EXPORT_IMG_COUNT; exportIndex++)
	{
		char filenameBuffer[128];
		int imageIndex = rand() % TrainingData.imageCount;
		sprintf_s(filenameBuffer, sizeof(filenameBuffer), "MNIST_Export_%d.bmp", exportIndex);

		if (!ExportMNISTImageAsBitmap(TrainingData.images[imageIndex], filenameBuffer))
		{
			break; // Break on failing to export one of the images.
		}

		printf("Exported image %d (label = %d) as bitmap file '%s'\n", imageIndex, TrainingData.labels[imageIndex], filenameBuffer);
	}

	system("pause");

	return 0;
}

int Main_TrainAndTest(int argc, char** argv)
{
	printf("Running Train and Test program...\n");

	MNIST_Dataset TrainingData = LoadTrainingSet();

	if (TrainingData.imageCount == 0)
	{
		printf("Failed to load training dataset.\n");
		return 1;
	}

	// Build training model from parameters.
	// TODO: Allow passing specific parameter to skip the Q&A code.

	printf("Creating new model...\n");

	int input_hiddenLayerCount = 0;
	int input_hiddenLayerSize = 0;

ENTER_PARAMETERS:
	printf("Specify number of Hidden Layers (N > 0): "); scanf_s("%d", &input_hiddenLayerCount);
	printf("Specify number of Neurons per Hidden Layer (N > 0): ");  scanf_s("%d", &input_hiddenLayerSize);

	// Check validity of provided parameters.
	if (input_hiddenLayerCount > 0 && input_hiddenLayerSize > 0)
	{
		goto PARAMETERS_ACCEPTED;
	}

INVALID_PARAMETERS:
	printf("Invalid model specifications. Please enter parameters within provided ranges.\n");
	goto ENTER_PARAMETERS;

PARAMETERS_ACCEPTED:
	
	printf("Generating model. Starting weights and biases will be initialized to random values and 0 respectively.\n");

	AIModel_NN newModel = NN_InitModel(input_hiddenLayerCount, input_hiddenLayerSize, true, true);

	if (newModel.layers == nullptr)
	{
		printf("Model generation failed. Aborting.\n");
		return 1;
	}
	
	printf("Successfully generated model. Performing feedforward on training set.\n");

	// TODO: It MIGHT be helpful to train it first (:
	for (int imageIndex = 0; imageIndex < TrainingData.imageCount; imageIndex++)
	{
		printf("FEEDFORWARD IMG %d (Label = %d). Results = \n", imageIndex, TrainingData.labels[imageIndex]);

		FeedforwardResult_NN result = NN_Feedforward_CPU(newModel, TrainingData.images[imageIndex]);

		printf("[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\n", result.values[0], result.values[1], result.values[2], result.values[3], result.values[4],
			result.values[5], result.values[6], result.values[7], result.values[8], result.values[9]);

		printf("PREDICTION = %d\n", result.GetHighestIndex());
	}

	system("pause");
	return 0;
}

int Main_TestModel(int argc, char** argv)
{
	printf("Running Test Model program...\n");

	printf("ERROR - UNIMPLEMENTED !\n");
	return 1;
}

/// <summary>
/// Main function. Redirects to one of the sub-main functions.
/// </summary>
/// <returns></returns>

int main(int argc, char** argv)
{
	system("cls");
	printf("Hello World !\n");

	if (argc == 1 || strcmp(argv[1], "help") == 0)
	{
		printf("This program can be ran in 3 modes, designed as a string argument in the CLA, followed by supported parameters:\n");
		printf("1 - test_export = displays some statistics about the dataset labels and exports a few images in .bmp format for manual checking.\n");
		printf("2 - train_and_test = Train AI then test it against the test dataset. Can accept extra parameters such as number of hidden layers, neurons per hidden layer and learning rate.\n");
		printf("3 - test_model = Use a pre-trained model and test it against the test dataset.");
		return 0;
	}

	if (strcmp(argv[1], "test_export") == 0)
	{
		return Main_TestExport(argc, argv);
	}

	if (strcmp(argv[1], "train_and_test") == 0)
	{
		return Main_TrainAndTest(argc, argv);
	}

	if (strcmp(argv[1], "test_model") == 0)
	{
		return Main_TestModel(argc, argv);
	}

	printf("Unable to parse argument '%s'. Use 'help' command for a list of possible arguments.", argv[0]);
	return 1;
}