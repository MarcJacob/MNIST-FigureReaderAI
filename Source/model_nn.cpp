// Implementation of functions specific to the Neural Network AI model.

#include "model_nn.h"
#include "string.h"
#include "intrin.h"

#define INPUT_LAYER_SIZE (MNIST_Img::IMG_WIDTH * MNIST_Img::IMG_HEIGHT)
#define OUTPUT_LAYER_SIZE (10)

#define RAND_BIAS_MIN ((neuron_bias)-2)
#define RAND_BIAS_MAX ((neuron_bias)2)

#define RAND_WEIGHT_MIN ((neuron_weight)-1)
#define RAND_WEIGHT_MAX ((neuron_weight)1)

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

AIModel_NN NN_InitModel(size_t hiddenLayerCount, size_t hiddenLayerSize, bool bRandomWeights, bool bRandomBiases)
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
	newModel.layers = (AIModel_NN::Layer**)modelMemory;											

	// Offset model memory to beginning of input layer.
	modelMemory += sizeof(AIModel_NN::Layer**) * newModel.layerCount;

	// Input layer.
	{
		size_t inputLayerStructOffset = sizeof(neuron_bias) * INPUT_LAYER_SIZE;

		// Allocate sizeof(AIModel_NN::Layer) after the input layer biases.
		newModel.layers[0] = (AIModel_NN::Layer*)(modelMemory + inputLayerStructOffset);		

		newModel.layers[0]->size = INPUT_LAYER_SIZE;

		// Allocate sizeof(neuron_bias) * INPUT_LAYER_SIZE for the input layer biases.
		newModel.layers[0]->biases = (neuron_bias*)(modelMemory);

		// Allocate sizeof(neuron_weight) * Hidden layer size for the input layer weights.
		newModel.layers[0]->weights = (neuron_weight*)(modelMemory + inputLayerStructOffset + sizeof(AIModel_NN::Layer));
	}

	// Offset model memory to beginning of first hidden layer.
	modelMemory += sizeof(neuron_bias) * INPUT_LAYER_SIZE + sizeof(AIModel_NN::Layer) + sizeof(neuron_weight) * INPUT_LAYER_SIZE * hiddenLayerSize;

	// Hidden layers, except for the last one (coming before the output layer).
	for (uint16_t layerIndex = 1; layerIndex < newModel.layerCount - 1; layerIndex++)
	{
		size_t layerStructOffset = sizeof(neuron_bias) * hiddenLayerSize;
		
		// Allocate sizeof(AIModel_NN::Layer) after the layer's biases.
		newModel.layers[layerIndex] = (AIModel_NN::Layer*)(modelMemory + layerStructOffset);

		newModel.layers[layerIndex]->size = hiddenLayerSize;

		// Allocate sizeof(neuron_bias) * Hidden layer size for the layer biases.
		newModel.layers[layerIndex]->biases = (neuron_bias*)(modelMemory);

		size_t nextLayerSize = hiddenLayerSize;
		if (layerIndex == newModel.layerCount - 2)
		{
			nextLayerSize = OUTPUT_LAYER_SIZE;
		}

		// Allocate sizeof(neuron_weight) * Layer size * Next layer size for the layer weights.
		newModel.layers[layerIndex]->weights = (neuron_weight*)(modelMemory + layerStructOffset + sizeof(AIModel_NN::Layer));

		// Offset model memory.
		modelMemory += sizeof(neuron_bias) * hiddenLayerSize + sizeof(AIModel_NN::Layer) + sizeof(neuron_weight) * hiddenLayerSize * nextLayerSize;
	}

	// Output layer.
	{
		size_t outputLayerStructOffset = sizeof(neuron_bias) * OUTPUT_LAYER_SIZE;

		// Allocate sizeof(AIModel_NN::Layer) after the output layer biases.
		newModel.layers[newModel.layerCount - 1] = (AIModel_NN::Layer*)(modelMemory + outputLayerStructOffset);

		newModel.layers[newModel.layerCount - 1]->size = OUTPUT_LAYER_SIZE;

		// Allocate sizeof(neuron_bias) * INPUT_LAYER_SIZE for the output layer biases.
		newModel.layers[newModel.layerCount - 1]->biases = (neuron_bias*)(modelMemory);

		// No weights !
		newModel.layers[newModel.layerCount - 1]->weights = nullptr;
	}

	// Offset the AI model memory.
	modelMemory += sizeof(neuron_bias) * OUTPUT_LAYER_SIZE + sizeof(AIModel_NN::Layer);

#if _DEBUG

	// Check that we've allocated exactly all of the model's memory.
	
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
			for (uint16_t neuronIndex = 0; neuronIndex < newModel.layers[layerIndex]->size; neuronIndex++)
			{
				newModel.layers[layerIndex]->biases[neuronIndex] = GenRandomBias();
			}
		}
		if (bRandomWeights && newModel.layers[layerIndex]->weights != nullptr) // Obviously only randomize the weights up until the second-to-last layer.
			// I'd rather have checked the layer index but Intellisense was "warning" me that dereferencing ->weights was dangerous unless I explicitly checked the pointer's
			// non-nullity. Thanks Microsoft.
		{
			for (uint16_t neuronIndex = 0; neuronIndex < newModel.layers[layerIndex]->size; neuronIndex++)
			{
				for (uint16_t targetNeuronIndex = 0; targetNeuronIndex < newModel.layers[layerIndex + 1]->size; targetNeuronIndex++)
				{
					newModel.layers[layerIndex]->weights[targetNeuronIndex * newModel.layers[layerIndex]->size + neuronIndex] = GenRandomWeight();
				}
			}
		}
	}

	return newModel;
}

// Defines an instance of a feedforward process for a Neural Network model. Contains only the activation values for each neurons.
// Must be initialized from and paired with a Model to work.
struct Feedforward_NN
{
	struct Layer
	{
		uint16_t size;
		neuron_activation values[1]; // Values array, of variable size, placed contiguously in memory after the structure itself.
	};

	Layer** layers; // Layer data (and by extension the whole structure) is allocated sequentially in memory with the format [Layer struct][values].
};

// Builds a feedforward instance for the given model. Allocates the required memory with malloc().
Feedforward_NN InitFeedforwardInstance(const AIModel_NN& Model)
{
	// Check that model is valid
	if (Model.layerCount < 3) // Input, Output and at least one hidden layer.
	{
		return {};
	}

	Feedforward_NN feedforward = {};
	
	// First pass: Get total neuron count from model to determine how much memory is required.
	size_t neuronCount = 0;
	for (int layerIndex = 0; layerIndex < Model.layerCount; layerIndex++)
	{
		neuronCount += Model.layers[layerIndex]->size;
	} 

	size_t memorySize = (
		sizeof(Feedforward_NN::Layer*) * Model.layerCount	// Pointer data for each layer structure, at the beginning of memory.
		+ sizeof(neuron_activation) * neuronCount 			// Activation values for all neurons.
		+ sizeof(Feedforward_NN::Layer) * Model.layerCount	// Structure data for each layer.
	);

	uint8_t* feedforwardMemory = (uint8_t*)malloc(memorySize);
	if (feedforwardMemory == nullptr)
	{
		return {};
	}

	memset(feedforwardMemory, 0, memorySize);

#if _DEBUG

	uint8_t* debug_feedforwardMemoryStart = feedforwardMemory;

#endif
	
	feedforward.layers = (Feedforward_NN::Layer**)feedforwardMemory;
	feedforwardMemory += sizeof(Feedforward_NN::Layer*) * Model.layerCount;

	// Second pass: Organize each layer, by assigning it the correct pointer value in layers, and giving it the correct size. 
	for (int layerIndex = 0; layerIndex < Model.layerCount; layerIndex++)
	{
		feedforward.layers[layerIndex] = (Feedforward_NN::Layer*)feedforwardMemory; // I don't know how to get rid of this warning. It is mathematically impossible for memorySize to be too small.
		feedforward.layers[layerIndex]->size = Model.layers[layerIndex]->size;

		// Offset memory pointer.
		feedforwardMemory += sizeof(Feedforward_NN::Layer) + feedforward.layers[layerIndex]->size * sizeof(neuron_activation);
	}

#if _DEBUG

	// Check that we've allocated exactly all of the feedforward structure's memory.

	size_t allocatedSize = feedforwardMemory - debug_feedforwardMemoryStart;
	if (allocatedSize != memorySize)
	{
		__debugbreak();
		return {};
	}

#endif

	return feedforward;
}

void InitializeInputLayer(Feedforward_NN& Feedforward, const MNIST_Img& InputImage)
{
	for (int inputNeuronIndex = 0; inputNeuronIndex < Feedforward.layers[0]->size; inputNeuronIndex++)
	{
		Feedforward.layers[0]->values[inputNeuronIndex] = InputImage.pixelValues[inputNeuronIndex] / (neuron_activation)256;
	}
}

// Performs feedforward on the passed initialized feedforward structure using the passed model.
void PerformFeedforward(Feedforward_NN& Feedforward, const AIModel_NN& Model)
{
	// For each layer starting at the first hidden layer, calculate the activation value for each neuron and apply a ReLU activation function on it.
	// If everything so far has been placed in memory correctly, then the read should be nearly perfectly sequential.
	for (int layerIndex = 1; layerIndex < Model.layerCount; layerIndex++)
	{
		for (int neuronIndex = 0; neuronIndex < Model.layers[layerIndex]->size; neuronIndex++)
		{
			neuron_activation& activationValue = Feedforward.layers[layerIndex]->values[neuronIndex];
			for (int sourceNeuronIndex = 0; sourceNeuronIndex < Model.layers[layerIndex - 1]->size; sourceNeuronIndex++)
			{
				activationValue += Feedforward.layers[layerIndex - 1]->values[sourceNeuronIndex] // Source value
					* Model.layers[layerIndex - 1]->weights[neuronIndex * Model.layers[layerIndex - 1]->size + sourceNeuronIndex]; // Source weight
			}

			activationValue += Model.layers[layerIndex]->biases[neuronIndex]; // Bias

			// ReLU
			activationValue = activationValue < 0 ? 0 : activationValue;
		}
	}
}

FeedforwardResult_NN ExtractResults(const Feedforward_NN& Feedforward, const AIModel_NN& Model)
{
	FeedforwardResult_NN Result = {};
	for (int outputNeuronIndex = 0; outputNeuronIndex < Feedforward.layers[Model.layerCount - 1]->size; outputNeuronIndex++)
	{
		Result.values[outputNeuronIndex] = Feedforward.layers[Model.layerCount - 1]->values[outputNeuronIndex];
	}

	return Result;
}

FeedforwardResult_NN NN_Feedforward_CPU(const AIModel_NN& Model, const MNIST_Img& Image)
{
	FeedforwardResult_NN Result = {};

	// Most stupid but functional way of doing caching so we don't re-init the feedforward structure every time.
	static AIModel_NN* LastModel = nullptr;
	static Feedforward_NN feedforward = {};
	if (&Model != LastModel)
	{
		// TODO: Check for compatibility between image and model ?

		feedforward = InitFeedforwardInstance(Model);
	}

	// Initialize the Model's first layer with the image's pixel values.
	InitializeInputLayer(feedforward, Image);

	// Perform feed forward / forward propagation.
	PerformFeedforward(feedforward, Model);

	return ExtractResults(feedforward, Model);
}

float NN_Train_CPU(AIModel_NN&, const MNIST_Dataset& Dataset, size_t StartImageIndex, size_t EndImageIndex)
{
	return 0.f;
}