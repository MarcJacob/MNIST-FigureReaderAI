// Contains symbol declarations for building and training Neural Network models using a MNIST dataset.

#include "mnist_dataset.h"

// MODEL: NEURAL NETWORK
// This sort of model attempts to solve the problem of classifying MNIST handwritten number images by making use of a Deep Neural Network based on an input layer made up of
// 28 * 28 neurons each corresponding to a pixel of the image, and 10 output neurons describing the "confidence" of the model that the correct label to associate to the input
// image is the corresponding digit (0-9 inclusive).
// The model has two main parameters on construction: Number of hidden layers, and number of neurons per hidden layer.

typedef float neuron_bias;
typedef float neuron_weight;

struct AIModel_NN
{
	struct Layer
	{
		uint16_t size; // Number of neurons in this layer.
		neuron_bias* biases; // Variable size array for the biases for this layer's neurons.
		neuron_weight* weights; // Variable size array for the weights going to the next layer. Weights are organized target-neuron-wise.
	};

	uint16_t layerCount;
	Layer** Layers; // Layer data is allocated sequentially in memory, with the following format: <Previous layer>...[BIASES_ARRAY][LAYER_STRUCT][WEIGHTS_ARRAY]...<Next Layer>
};

// Initializes a new NN model. Necessary heap memory is allocated using malloc().
AIModel_NN InitializeNewModel(size_t hiddenLayerCount, size_t hiddenLayerSize, bool bRandomWeights = true, bool bRandomBiases = false);