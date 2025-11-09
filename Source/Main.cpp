#include <stdio.h>
#include <stdint.h>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include "mnist.h"

/*
	MNIST Dataset-based figure reading AI.
	The AI is written entirely using no libraries other than default Windows & C / C++ libraries and CUDA.
*/

#define PATH_TRAINING_SET_LABELS "D:\\Datasets\\MNIST\\train-labels.idx1-ubyte"
#define PATH_TRAINING_SET_IMAGES "D:\\Datasets\\MNIST\\train-images.idx3-ubyte"

#define PATH_TEST_SET_LABELS "D:\\Datasets\\MNIST\\t10k-labels.idx1-ubyte"
#define PATH_TEST_SET_IMAGES "D:\\Datasets\\MNIST\\t10k-images.idx3-ubyte"

#define KILOBYTES(n) (1024 * n)

int Main_TestExport(int argc, char** argv)
{
	printf("Running Test Export program...\n");

	// Just load the entire dataset in memory, it's not that large.
	MNIST_Dataset TrainingData;

	uint8_t* trainingLabelsBuffer = (uint8_t*)malloc(KILOBYTES(60));
	uint8_t* trainingImagesBuffer = (uint8_t*)malloc(KILOBYTES(46000));

	if (trainingLabelsBuffer == nullptr || trainingImagesBuffer == nullptr)
	{
		printf("Error: Out of memory for Dataset file reading buffers.\n");
		return 1;
	}

	// TRAINING LABELS
	{
		HANDLE file_trainingLabels = CreateFile(PATH_TRAINING_SET_LABELS, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
		if (file_trainingLabels == INVALID_HANDLE_VALUE)
		{
			printf("Error opening Training Labels file '%s'. Error Code = %d\n", PATH_TRAINING_SET_LABELS, GetLastError());
			return 1;
		}


		DWORD readBytes = 0;
		if (!ReadFile(file_trainingLabels, trainingLabelsBuffer, KILOBYTES(60), &readBytes, NULL))
		{
			printf("Error reading Training Labels file '%s'. Error Code = %d\n", PATH_TRAINING_SET_LABELS, GetLastError());
			CloseHandle(file_trainingLabels);
			return 1;
		}

		CloseHandle(file_trainingLabels);
	}

	// TRAINING IMAGES
	{
		HANDLE file_trainingImages = CreateFile(PATH_TRAINING_SET_IMAGES, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
		if (file_trainingImages == INVALID_HANDLE_VALUE)
		{
			printf("Error opening Training Images file '%s'. Error Code = %d\n", PATH_TRAINING_SET_IMAGES, GetLastError());
			return 1;
		}


		DWORD readBytes = 0;
		if (!ReadFile(file_trainingImages, trainingImagesBuffer, KILOBYTES(46000), &readBytes, NULL))
		{
			printf("Error reading Training Images file '%s'. Error Code = %d\n", PATH_TRAINING_SET_IMAGES, GetLastError());
			CloseHandle(file_trainingImages);
			return 1;
		}

		CloseHandle(file_trainingImages);
	}

	TrainingData = ParseDataset(trainingLabelsBuffer, trainingImagesBuffer);

	// Verify integrity of read data
	if (TrainingData.imageCount == 0 || TrainingData.imageCount != TrainingData.labelCount)
	{
		printf("Error: Training Image Count does not match Training Label Count (%d != %d), or failed to load appropriately !\n", TrainingData.imageCount, TrainingData.labelCount);
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

	printf("ERROR - UNIMPLEMENTED !\n");
	return 1;
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