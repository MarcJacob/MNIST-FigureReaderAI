// Contains symbol declarations for building and training Neural Network models using a MNIST dataset.

#include "mnist_dataset.h"

// MODEL: NEURAL NETWORK
// This sort of model attempts to solve the problem of classifying MNIST handwritten number images by making use of a Deep Neural Network based on an input layer made up of
// 28 * 28 neurons each corresponding to a pixel of the image, and 10 output neurons describing the "confidence" of the model that the correct label to associate to the input
// image is the corresponding digit (0-9 inclusive).
// The model has two main parameters on construction: Number of hidden layers, and number of neurons per hidden layer.

typedef float neuron_bias;
typedef float neuron_weight;
typedef float neuron_activation;

struct AIModel_NN
{
	static constexpr int OUTPUT_VALUE_COUNT = 10; // The NN Model outputs 10 values, one for its confidence in the image being the associated digit.

	struct Layer
	{
		uint16_t size; // Number of neurons in this layer.
		neuron_bias* biases; // Variable size array for the biases for this layer's neurons.
		neuron_weight* weights; // Variable size array for the weights going to the next layer. Weights are organized target-neuron-wise.
	};

	uint16_t layerCount;
	size_t modelMemorySize; // Room this model takes in memory, in bytes.
	Layer** layers; // Layer data is allocated sequentially in memory, with the following format: <Previous layer>...[BIASES_ARRAY][LAYER_STRUCT][WEIGHTS_ARRAY]...<Next Layer>
};

struct FeedforwardResult_NN
{
	neuron_activation values[AIModel_NN::OUTPUT_VALUE_COUNT];
	int GetHighestIndex() const { 
		int maxIndex = 0; 
		for (int i = 0; i < AIModel_NN::OUTPUT_VALUE_COUNT; i++) 
			if (values[i] > values[maxIndex]) 
				maxIndex = i; 
		return maxIndex;
	}
};

// Initializes a new NN model. Necessary heap memory is allocated using malloc().
AIModel_NN NN_InitModel(size_t hiddenLayerCount, size_t hiddenLayerSize, bool bRandomWeights = true, bool bRandomBiases = false);

// Frees the memory taken up by a Neural Network model.
void NN_FreeModel(AIModel_NN& Model);

// Performs a single instance of Feed Forward using the passed model and image as input on the CPU.
FeedforwardResult_NN NN_Feedforward_CPU(const AIModel_NN& Model, const MNIST_Img& Image);

// Performs an "epoch" of training over the specified dataset from the start image index to the end image index.
// The model accumulates "Error" as it attempts to process each image and compares its predictions to the associated label. Once the epoch is ran,
// Backpropagation is performed to tune the model in order to reduce future error.
// The entire process runs on the CPU.
// Returns the accumulated error value over the outputs during the epoch.
float NN_Train_CPU(AIModel_NN&, const MNIST_Dataset& Dataset, size_t StartImageIndex, size_t EndImageIndex);