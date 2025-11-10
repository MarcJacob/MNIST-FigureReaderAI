// Implementation of functions specific to the Neural Network AI model.

#include "model_nn.h"
#include "string.h"
#include "intrin.h"

#define INPUT_LAYER_SIZE (MNIST_Img::IMG_WIDTH * MNIST_Img::IMG_HEIGHT)
#define OUTPUT_LAYER_SIZE (10)

#define RAND_BIAS_MIN ((neuron_bias)-2)
#define RAND_BIAS_MAX ((neuron_bias)2)

#define RAND_WEIGHT_MIN ((neuron_weight)-2)
#define RAND_WEIGHT_MAX ((neuron_weight)2)

neuron_bias GenRandomBias()
{
	// Let's keep it really simple. Quantize whatever rand() returns as a thousandth and use that.
	int32_t randGen = rand() % ((size_t)(RAND_BIAS_MAX - RAND_BIAS_MIN) * 1000) + RAND_BIAS_MIN * 1000;
	return randGen * (neuron_bias)0.001;
}

neuron_weight GenRandomWeight()
{
	// Let's keep it really simple. Quantize whatever rand() returns as a thousandth and use that.
	int32_t randGen = rand() % (size_t)((RAND_BIAS_MAX - RAND_BIAS_MIN) * 1000) + RAND_BIAS_MIN * 1000;
	return randGen * (neuron_weight)0.001;
}

AIModel_NN InitializeNewModel(size_t hiddenLayerCount, size_t hiddenLayerSize, bool bRandomWeights, bool bRandomBiases)
{
	// Check inputs. Return empty model as a signal that input parameters are invalid.
	if (hiddenLayerCount == 0 || hiddenLayerSize == 0)
	{
		return {};
	}

	AIModel_NN newModel = {};
	newModel.layerCount = hiddenLayerCount + 2; // Hidden layer count + input + output.

	// Estimate total memory size of the model and allocate one chunk of memory for it.
	// This might stop working with large models but that shouldn't be necessary here so let's optimize memory accesses.
	// An alternative function could be easily made to split the data into multiple chunks without changing the model structure.
	// This allocation strategy also has the major drawback of being quite painful to adapt to adding extra struff in the future, but I don't expect this will happen.

	size_t modelMemorySize = (
		sizeof(AIModel_NN::Layer*) * newModel.layerCount											// Pointers to layer structures.
		+ sizeof(AIModel_NN::Layer) * newModel.layerCount											// Layer structures.
		+ sizeof(neuron_bias) * INPUT_LAYER_SIZE													// Biases for input layer.
		+ sizeof(neuron_weight) * INPUT_LAYER_SIZE * hiddenLayerSize								// Weights for input layer to first hidden layer.
		+ sizeof(neuron_bias) * hiddenLayerSize * hiddenLayerCount									// Biases for hidden layers.
		+ sizeof(neuron_weight) * hiddenLayerSize * hiddenLayerSize * (hiddenLayerCount - 1)		// Weights for hidden layers to next hidden layer, except last one.
		+ sizeof(neuron_weight) * hiddenLayerSize * OUTPUT_LAYER_SIZE								// Weights for last hidden layer to output layer.
		+ sizeof(neuron_bias) * OUTPUT_LAYER_SIZE													// Biases for output layer. No weights.
	);

	uint8_t* modelMemory = (uint8_t*)malloc(modelMemorySize);
	
#if _DEBUG
	uint8_t* debug_modelMemoryStart = modelMemory;
#endif

	if (modelMemory == nullptr)
	{
		return {};
	}

	// Zero out the entire model.
	memset(modelMemory, 0, modelMemorySize);

	// Now it is a matter of allocating that memory to the layers of the model appropriately, following this format:
	// [LAYER_BIASES][LAYER_STRUCT][LAYER_WEIGHTS]
	// for every layer.

	// Allocate sizeof(AIModel_NN::Layer*) * total layer count
	newModel.Layers = (AIModel_NN::Layer**)modelMemory;											

	// Offset model memory to beginning of input layer.
	modelMemory += sizeof(AIModel_NN::Layer**) * newModel.layerCount;

	// Input layer.
	{
		size_t inputLayerStructOffset = sizeof(neuron_bias) * INPUT_LAYER_SIZE;

		// Allocate sizeof(AIModel_NN::Layer) after the input layer biases.
		newModel.Layers[0] = (AIModel_NN::Layer*)(modelMemory + inputLayerStructOffset);		

		newModel.Layers[0]->size = INPUT_LAYER_SIZE;

		// Allocate sizeof(neuron_bias) * INPUT_LAYER_SIZE for the input layer biases.
		newModel.Layers[0]->biases = (neuron_bias*)(modelMemory);

		// Allocate sizeof(neuron_weight) * Hidden layer size for the input layer weights.
		newModel.Layers[0]->weights = (neuron_weight*)(modelMemory + inputLayerStructOffset + sizeof(AIModel_NN::Layer));
	}

	// Offset model memory to beginning of first hidden layer.
	modelMemory += sizeof(neuron_bias) * INPUT_LAYER_SIZE + sizeof(AIModel_NN::Layer) + sizeof(neuron_weight) * INPUT_LAYER_SIZE * hiddenLayerSize;

	// Hidden layers, except for the last one (coming before the output layer).
	for (uint16_t layerIndex = 1; layerIndex < newModel.layerCount - 1; layerIndex++)
	{
		size_t layerStructOffset = sizeof(neuron_bias) * hiddenLayerSize;
		
		// Allocate sizeof(AIModel_NN::Layer) after the layer's biases.
		newModel.Layers[layerIndex] = (AIModel_NN::Layer*)(modelMemory + layerStructOffset);

		newModel.Layers[layerIndex]->size = hiddenLayerSize;

		// Allocate sizeof(neuron_bias) * Hidden layer size for the layer biases.
		newModel.Layers[layerIndex]->biases = (neuron_bias*)(modelMemory);

		size_t nextLayerSize = hiddenLayerSize;
		if (layerIndex == newModel.layerCount - 2)
		{
			nextLayerSize = OUTPUT_LAYER_SIZE;
		}

		// Allocate sizeof(neuron_weight) * Layer size * Next layer size for the layer weights.
		newModel.Layers[layerIndex]->weights = (neuron_weight*)(modelMemory + layerStructOffset + sizeof(AIModel_NN::Layer));

		// Offset model memory.
		modelMemory += sizeof(neuron_bias) * hiddenLayerSize + sizeof(AIModel_NN::Layer) + sizeof(neuron_weight) * hiddenLayerSize * nextLayerSize;
	}

	// Output layer.
	{
		size_t outputLayerStructOffset = sizeof(neuron_bias) * OUTPUT_LAYER_SIZE;

		// Allocate sizeof(AIModel_NN::Layer) after the output layer biases.
		newModel.Layers[newModel.layerCount - 1] = (AIModel_NN::Layer*)(modelMemory + outputLayerStructOffset);

		newModel.Layers[newModel.layerCount - 1]->size = OUTPUT_LAYER_SIZE;

		// Allocate sizeof(neuron_bias) * INPUT_LAYER_SIZE for the output layer biases.
		newModel.Layers[newModel.layerCount - 1]->biases = (neuron_bias*)(modelMemory);

		// No weights !
		newModel.Layers[newModel.layerCount - 1]->weights = nullptr;
	}

	// Offset the AI model memory.
	modelMemory += sizeof(neuron_bias) * OUTPUT_LAYER_SIZE + sizeof(AIModel_NN::Layer);

#if _DEBUG

	// Check that we've allocated all of the model's memory.
	size_t allocatedSize = modelMemory - debug_modelMemoryStart;
	if (allocatedSize != modelMemorySize)
	{
		__debugbreak();
		return {};
	}

#endif

	// The model has been allocated. If requested, let's randomize some values !
	if (bRandomBiases || bRandomWeights)
	for (uint16_t layerIndex = 0; layerIndex < newModel.layerCount; layerIndex++) // Go through the model memory linearly (biases then weights) to take advantage of the cache.
	{
		if (bRandomBiases)
		{
			for (uint16_t neuronIndex = 0; neuronIndex < newModel.Layers[layerIndex]->size; neuronIndex++)
			{
				newModel.Layers[layerIndex]->biases[neuronIndex] = GenRandomBias();
			}
		}
		if (bRandomWeights && newModel.Layers[layerIndex]->weights != nullptr) // Obviously only randomize the weights up until the second-to-last layer.
			// I'd rather have checked the layer index but Intellisense was "warning" me that dereferencing ->weights was dangerous unless I explicitly checked the pointer's
			// non-nullity. Thanks Microsoft.
		{
			for (uint16_t neuronIndex = 0; neuronIndex < newModel.Layers[layerIndex]->size; neuronIndex++)
			{
				for (uint16_t targetNeuronIndex = 0; targetNeuronIndex < newModel.Layers[layerIndex + 1]->size; targetNeuronIndex++)
				{
					newModel.Layers[layerIndex]->weights[targetNeuronIndex * newModel.Layers[layerIndex]->size + neuronIndex] = GenRandomWeight();
				}
			}
		}
	}

	return newModel;
}