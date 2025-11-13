// Implementation of functions specific to the Neural Network AI model.

#include "model_nn.h"
#include "string.h"
#include "intrin.h"

#include <math.h>

#include <stdio.h>

#define INPUT_LAYER_SIZE (MNIST_Img::IMG_WIDTH * MNIST_Img::IMG_HEIGHT)
#define OUTPUT_LAYER_SIZE (AIModel_NN::OUTPUT_VALUE_COUNT)

neuron_bias GenRandomBias(neuron_bias Min, neuron_bias Max)
{
	// Let's keep it really simple. Quantize whatever rand() returns as a thousandth and use that.
	int32_t randGen = rand() % (int32_t)((Max - Min) * 1000);
	return (randGen * (neuron_bias)0.001) + Min;
}

neuron_weight GenRandomWeight(neuron_weight Min, neuron_weight Max)
{
	// Let's keep it really simple. Quantize whatever rand() returns as a thousandth and use that.
	int32_t randGen = rand() % (int32_t)((Max - Min) * 1000);
	return (randGen * (neuron_weight)0.001) + Min;
}

AIModel_NN NN_InitModel(uint16_t hiddenLayerCount, uint16_t hiddenLayerSize, bool bRandomWeights, bool bRandomBiases)
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
	newModel.modelMemorySize = modelMemorySize;

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
				newModel.layers[layerIndex]->biases[neuronIndex] = GenRandomBias(-0.2f, 0.2f);
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
					newModel.layers[layerIndex]->weights[targetNeuronIndex * newModel.layers[layerIndex]->size + neuronIndex] = GenRandomWeight(-4.f / newModel.layers[layerIndex]->size, 4.f / newModel.layers[layerIndex]->size);
				}
			}
		}
	}

	return newModel;
}

void NN_FreeModel(AIModel_NN& Model)
{
	// The Model's entire memory is tied to a single allocation at the location of its layers pointer.
	free(Model.layers);
}

// MODEL CORE FUNCTIONS

// Activation Function : "Leaky" ReLU with slight (not flat) gradient below 0.
inline neuron_activation Activation(neuron_activation InActivation)
{
	return InActivation < 0 ? __max(-10000, InActivation * 0.1f) : __min(10000, InActivation);
}

// Activation Function Derivative: "Leaky" ReLU with 0.1 gradient below 0 and 1 at and above.
inline neuron_activation d_Activation(neuron_activation InActivation)
{
	return InActivation < 0 ? 0.1 : 1;
}

// Returns the cost of a neuron's activation compared to what is desired.
inline model_cost Cost(neuron_activation Activation, neuron_activation Desired)
{
	return (Activation - Desired) * (Activation - Desired);
}

// Returns the cost gradient of a source neuron's weight.
inline model_cost GetWeightCostGradient(neuron_activation SourceNeuronActivation, neuron_activation TargetNeuronActivation, neuron_activation DesiredActivation)
{
	// Note: Using TargetNeuronActivation in d_Activation here isn't quite correct because we need the neuron value PRE activation function, but since we are using ReLU it should work.
	return SourceNeuronActivation * 2 * (TargetNeuronActivation - DesiredActivation) * d_Activation(TargetNeuronActivation);
}

// Returns the cost gradient of a neuron's bias.
inline model_cost GetBiasCostGradient(neuron_activation TargetNeuronActivation, neuron_activation DesiredActivation)
{
	return 2 * (TargetNeuronActivation - DesiredActivation) * d_Activation(TargetNeuronActivation);
}

// Returns the cost gradient of a source neuron's value from the perspective of a single target neuron. Take care to average together all the values for all
// target neurons on the target layer for each source neuron !
inline model_cost GetSinglePathNeuronCostGradient(neuron_weight SourceWeight, neuron_activation TargetNeuronActivation, neuron_activation DesiredActivation)
{
	return SourceWeight * 2 * (TargetNeuronActivation - DesiredActivation) * d_Activation(TargetNeuronActivation);
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

	size_t layerCount;
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
	feedforward.layerCount = Model.layerCount;

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
		feedforward.layers[layerIndex] = (Feedforward_NN::Layer*)feedforwardMemory;
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

void FreeFeedforwardInstance(Feedforward_NN& Feedforward)
{
	// The entire memory of the Feedforward structure is tied to a single allocation at wherever the layers pointer is pointing.
	free(Feedforward.layers);
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
			activationValue = 0;
			for (int sourceNeuronIndex = 0; sourceNeuronIndex < Model.layers[layerIndex - 1]->size; sourceNeuronIndex++)
			{
				activationValue += Feedforward.layers[layerIndex - 1]->values[sourceNeuronIndex] // Source value
					* Model.layers[layerIndex - 1]->weights[neuronIndex * Model.layers[layerIndex - 1]->size + sourceNeuronIndex]; // Source weight
			}

			activationValue += Model.layers[layerIndex]->biases[neuronIndex]; // Bias
			activationValue = Activation(activationValue);
		}
	}

#if 0 // TODO Determine if this is really necessary. Having this happen on the output layer makes the determination of a learning gradient far more complicated.
	// Since we want the result values to be a distribution of probabilities, we need the sum of all outputs to be equal to 1 and between 0 and 1.
	// We also want lower output values to be closer to 0 and higher output values to be closer to 1.
	// The simplest way of achieving this seems to be to raise something to the power of the outputs ("Transformation", bringing all outputs into the positive space),
	// then computing the sum of those transformed values and dividing each transformed value by that sum ("Normalization", bringing all outputs into the [0, 1] space).
	// The result will have the properties we need: Greater or equal to 0, Lower or equal to 1, and with lower output values trending towards 0 and higher output values trending
	// towards 1.
	// Raising something to the power of the outputs also has the interesting property of making the "transformed sum" never equal 0 !

	// Find maximum for first normalization pass.
	// It turns out this is necessary to prevent very high output values from breaking the algorithm.
	neuron_activation maxOutput = Feedforward.layers[Model.layerCount - 1]->values[0];
	for (int outputNeuronIndex = 1; outputNeuronIndex < OUTPUT_LAYER_SIZE; outputNeuronIndex++)
	{
		neuron_activation output = Feedforward.layers[Model.layerCount - 1]->values[outputNeuronIndex];
		if (output > maxOutput)
		{
			maxOutput = output;
		}
	}

	// Compute transformed values and their sum.
	double_t transformedSum = 0;
	for (int outputNeuronIndex = 0; outputNeuronIndex < OUTPUT_LAYER_SIZE; outputNeuronIndex++)
	{
		// Raise exponential to the value of the output normalized by maximum for greater number (and system) stability.
		Feedforward.layers[Model.layerCount - 1]->values[outputNeuronIndex] = exp(Feedforward.layers[Model.layerCount - 1]->values[outputNeuronIndex] / maxOutput);
		transformedSum += Feedforward.layers[Model.layerCount - 1]->values[outputNeuronIndex];
	}

	// Compute normalized transformed values.
	for (int outputNeuronIndex = 0; outputNeuronIndex < OUTPUT_LAYER_SIZE; outputNeuronIndex++)
	{
		Feedforward.layers[Model.layerCount - 1]->values[outputNeuronIndex] /= transformedSum;
	}
#endif
}

// Read the output values of a feedforward structure and convert them to a valid probability distribution within a Result structure.
FeedforwardResult_NN ExtractResults(const Feedforward_NN& Feedforward)
{
	FeedforwardResult_NN Result = {};

	// Read from the feedforward structure's last layer and return.
	for (int outputNeuronIndex = 0; outputNeuronIndex < OUTPUT_LAYER_SIZE; outputNeuronIndex++)
	{
		Result.values[outputNeuronIndex] = Feedforward.layers[Feedforward.layerCount - 1]->values[outputNeuronIndex];
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
		if (LastModel != nullptr)
		{
			// Free previous Feed Forward structure.
			FreeFeedforwardInstance(feedforward);
			feedforward = {};
		}

		// TODO: Check for compatibility between image and model ?

		feedforward = InitFeedforwardInstance(Model);
	}

	// Initialize the Model's first layer with the image's pixel values.
	InitializeInputLayer(feedforward, Image);

	// Perform feed forward / forward propagation.
	PerformFeedforward(feedforward, Model);

	return ExtractResults(feedforward);
}

// Structure that stores requested changes to neuron weights and biases on each layer (except input) for a certain number of iterations.
// Once filled in, it can be applied to the model used to generate it by averaging all the changes together accross all iterations and
// using those values to perform the actual model change.
struct Learning_NN
{
	struct Layer
	{
		// Size of previous layer, which determines the size of the weight changes.
		size_t previousLayerSize;
		// Size of the layer itself, which determines the size of the weight changes and the bias changes.
		size_t layerSize;
		// Requested weight changes, size equals (previous layer size) * (layer size) * sizeof(neuron_weight) * (batch size)
		// Weight changes are organized iteration-wise, then target-neuron-wise (since these are the values that "belong" to the same layer element).
		neuron_weight* weightChanges;
		// Requested bias changes, size equals (layer size) * sizeof(neuron_bias) * (batch size)
		// Bias changes are organized iteration-wise.
		neuron_bias* biasChanges;
	};

	size_t batchSize; // Expected size of the full "learning batch". For each iteration in the batch, there should be requested change for every weight and bias.
	size_t layerCount;
	Layer** layers; // Main layers memory, with the following format: [layers]...[Layer N-1 struct][Weight Changes][Bias changes][Layer N]...
};

Learning_NN InitLearningInstance(const AIModel_NN& Model, size_t BatchSize)
{
	Learning_NN newLearning = {};

	// Check that model is valid
	if (Model.layerCount < 3) // Input, Output and at least one hidden layer.
	{
		return {};
	}

	// Check that we are asking for at least one iteration per batch.
	if (BatchSize == 0)
	{
		return {};
	}

	newLearning.layerCount = Model.layerCount - 1; // No input layer.
	newLearning.batchSize = BatchSize;
	size_t learningMemorySize = sizeof(Learning_NN::Layer*) * newLearning.layerCount; // Layer pointers.

	// Each layer has a specific size since their internal data size depends on their size in the model and the size of the previous layer.
	for (int layerIndex = 0; layerIndex < newLearning.layerCount; layerIndex++)
	{
		int modelLayerIndex = layerIndex + 1;
		learningMemorySize += sizeof(Learning_NN::Layer) // Layer structure
			+ Model.layers[modelLayerIndex - 1]->size * Model.layers[modelLayerIndex]->size * sizeof(neuron_weight) // Weights from previous layer to this one
			+ Model.layers[modelLayerIndex]->size * sizeof(neuron_bias); // Biases for this layer
	}

	uint8_t* learningMemory = (uint8_t*)malloc(learningMemorySize);
	newLearning.layers = (Learning_NN::Layer**)learningMemory;
	if (newLearning.layers == nullptr)
	{
		return {};
	}

	memset(learningMemory, 0, learningMemorySize);

#if _DEBUG

	uint8_t* learningMemoryStart = learningMemory;

#endif

	// Offset memory pointer.
	learningMemory += sizeof(Learning_NN::Layer*) * newLearning.layerCount;

	// Organize layers.
	for (int layerIndex = 0; layerIndex < newLearning.layerCount; layerIndex++)
	{
		int modelLayerIndex = layerIndex + 1;

		newLearning.layers[layerIndex] = (Learning_NN::Layer*)learningMemory;
		newLearning.layers[layerIndex]->layerSize = Model.layers[modelLayerIndex]->size;
		newLearning.layers[layerIndex]->previousLayerSize = Model.layers[modelLayerIndex - 1]->size;

		newLearning.layers[layerIndex]->weightChanges = (neuron_weight*)(learningMemory + sizeof(Learning_NN::Layer));
		newLearning.layers[layerIndex]->biasChanges = (neuron_bias*)(
			learningMemory + sizeof(Learning_NN::Layer)
			+ Model.layers[modelLayerIndex - 1]->size * Model.layers[modelLayerIndex]->size * sizeof(neuron_weight));

		// Offset memory pointer.
		learningMemory += sizeof(Learning_NN::Layer)
			+ Model.layers[modelLayerIndex - 1]->size * Model.layers[modelLayerIndex]->size * sizeof(neuron_weight)
			+ Model.layers[modelLayerIndex]->size * sizeof(neuron_bias);
	}

#if _DEBUG

	// Check that we've allocated exactly the expected amount of memory.
	size_t allocatedSize = learningMemory - learningMemoryStart;
	if (allocatedSize != learningMemorySize)
	{
		__debugbreak();
	}

#endif

	return newLearning;
}

void FreeLearningInstance(Learning_NN& Learn)
{
	// All the allocated memory for a Learn structure is tied to its layers pointer.
	free(Learn.layers);
}

// Determines the Output Error of the given Feedforward result against the input Image and Label, and fills in the passed Learning Structure at the relevant iteration.
// Returns the total Error "processed" for this iteration.
float PerformBackpropagation(const AIModel_NN& Model, const Feedforward_NN& Feedforward, const MNIST_Img& InputImage, const int8_t& InputLabel, Learning_NN& Learning, size_t IterationIndex)
{
	// Check that label value is valid.
	if (InputLabel > 9) return -1.f;

	// Check that passed in Learning Structure is large enough for the Iteration Index.
	if (Learning.batchSize <= IterationIndex)
	{
		return -1.f;
	}

	// Use a double-buffered system for holding Desired Values for the current and previous layer while backpropagating.
	// They are as large as the largest layer in the model, meaning no bound checking or extra allocations are ever required for them in the whole backpropagation process.
	neuron_activation* currentDesiredValues = nullptr;
	neuron_activation* previousLayerValueGradients = nullptr;
	size_t desiredValuesBufferSize;
	
	// Use the largest layer size in the model to determine the allocation size of the buffers, with the exception of the input buffer
	// for which "desired values" are irrelevant.

	uint16_t largestLayerSize = 0;
	for (int layerIndex = 1; layerIndex < Model.layerCount; layerIndex++)
	{
		if (Model.layers[layerIndex]->size > largestLayerSize)
		{
			largestLayerSize = Model.layers[layerIndex]->size;
		}
	}

	if (largestLayerSize == 0)
	{
		return -1.f;
	}

	desiredValuesBufferSize = largestLayerSize * sizeof(neuron_activation);
	currentDesiredValues = (neuron_activation*)malloc(desiredValuesBufferSize);
	previousLayerValueGradients = (neuron_activation*)malloc(desiredValuesBufferSize);

	if (currentDesiredValues == nullptr || previousLayerValueGradients == nullptr)
	{
		return -1.f;
	}

	memset(currentDesiredValues, 0, desiredValuesBufferSize);
	memset(previousLayerValueGradients, 0, desiredValuesBufferSize);
	

	// Initialize the output layer desired values according to the label.

	currentDesiredValues[InputLabel] = 1;

	// Calculate total output cost. The goal of Backpropagation is to lower this value.
	double totalOutputCost = 0.f;

	// Add to the total error if on output layer.
	for (int outputNeuronIndex = 0; outputNeuronIndex < OUTPUT_LAYER_SIZE; outputNeuronIndex++)
	{
		neuron_activation output = Feedforward.layers[Model.layerCount - 1]->values[outputNeuronIndex];
		double cost = Cost(output, currentDesiredValues[outputNeuronIndex]);
		totalOutputCost += cost;
	}

	// Backpropagation
	// Work our way backwards starting from the output layer.
	for (int layerIndex = Model.layerCount - 1; layerIndex > 0; layerIndex--)
	{
		// Learning Layer in which to record the desired changes in weights and biases.
		Learning_NN::Layer& learningLayer = *Learning.layers[layerIndex - 1];

		// For each neuron in the layer, determine the error between the corresponding FeedforwardResult value and desired value.
		// Then, determine a change in inbound weight values and bias to reduce this error.
		// Determine the relative error for all inbound neurons and add it to their place in the previous layer desired values array.
		// Those sums will then be converted back to an actual desired value as a post processing step using their actual value during the feed forward phase, saving memory.
		for (int neuronIndex = 0; neuronIndex < Model.layers[layerIndex]->size; neuronIndex++)
		{
			neuron_activation outputActivation = Feedforward.layers[layerIndex]->values[neuronIndex];
			neuron_activation outputCost = Cost(outputActivation, currentDesiredValues[neuronIndex]);

			// Bias gradient descent
			learningLayer.biasChanges[neuronIndex] += -GetBiasCostGradient(outputActivation, outputCost);

			for (int sourceNeuronIndex = 0; sourceNeuronIndex < Model.layers[layerIndex - 1]->size; sourceNeuronIndex++)
			{
				// Weight gradient descent
				neuron_activation sourceActivation = Feedforward.layers[layerIndex - 1]->values[sourceNeuronIndex];
				neuron_weight weightChange = -GetWeightCostGradient(sourceActivation, outputActivation, currentDesiredValues[neuronIndex]);
				learningLayer.weightChanges[neuronIndex * Model.layers[layerIndex - 1]->size + sourceNeuronIndex]
					+= weightChange;
			}

			// Determine cost gradient for the overall activation value of each source neuron. Add them into Previous Layer Desired Value Gradients where they'll be converted
			// back to a Desired Value once the current layer has completed back propagation.
			if (layerIndex > 1)
			for (int sourceNeuronIndex = 0; sourceNeuronIndex < Model.layers[layerIndex - 1]->size; sourceNeuronIndex++)
			{
				neuron_weight sourceWeight = Model.layers[layerIndex - 1]->weights[neuronIndex * Model.layers[layerIndex - 1]->size + sourceNeuronIndex];
				previousLayerValueGradients[sourceNeuronIndex] += -GetSinglePathNeuronCostGradient(sourceWeight, outputActivation, currentDesiredValues[neuronIndex]);
			}
		}

		// Post Process the Previous Layer Value Gradients buffer - it for now contains the sum of the single path gradients.
		// Turn the sum into an average by dividing it by the number of neurons in the current layer then offset it by actual activation value of each source neuron
		// to turn it into a desired value for them.
		if (layerIndex > 1)
		for (int sourceNeuronIndex = 0; sourceNeuronIndex < Model.layers[layerIndex - 1]->size; sourceNeuronIndex++)
		{
			// Average out the gradient sum, which should output the "Overall best" gradient for the overall cost of the current layer.
			// Effectively this means weights and biases adapt to lower the cost of individual neurons while the neurons seek to lower the cost they add to the next layer.
			previousLayerValueGradients[sourceNeuronIndex] /= Model.layers[layerIndex]->size;

			// Get the desired value for the source neuron by offsetting the average gradient by the source neuron's activation value.
			previousLayerValueGradients[sourceNeuronIndex] += Feedforward.layers[layerIndex - 1]->values[sourceNeuronIndex];
		}

		// Switch the desired value buffers.
		neuron_activation* swap = currentDesiredValues;
		currentDesiredValues = previousLayerValueGradients;
		previousLayerValueGradients = swap;

		// Zero out previous layer buffer to make sure no unrelated information survives to the next iteration.
		memset(previousLayerValueGradients, 0, desiredValuesBufferSize);
	}

	// Free allocated desired value buffers.
	free(currentDesiredValues);
	free(previousLayerValueGradients);

	return totalOutputCost;
}

// Applies the accumulated changes in a Learning structure onto its origin model by averaging the sum of the requested changes
// collected during backpropagation and applying them to the Model
void ApplyLearning(const Learning_NN& Learn, AIModel_NN& Model, float LearningRate)
{
	// Go through every neuron in the model starting from the first hidden layer, aggregate the Learning changes and apply them.

	for (int layerIndex = Model.layerCount - 1; layerIndex > 0; layerIndex--)
	{
		AIModel_NN::Layer& currentLayer = *Model.layers[layerIndex];
		AIModel_NN::Layer& previousLayer = *Model.layers[layerIndex - 1];
		Learning_NN::Layer& learningLayer = *Learn.layers[layerIndex - 1]; // Offset from Model to Learning layer.
		for (int neuronIndex = 0; neuronIndex < currentLayer.size; neuronIndex++)
		{
			for (int sourceNeuronIndex = 0; sourceNeuronIndex < previousLayer.size; sourceNeuronIndex++)
			{
				// Aggregate the weight changes by averaging the sum value using the batch size and modify the model accordingly.
				neuron_weight changeSum = learningLayer.weightChanges[neuronIndex * learningLayer.layerSize + sourceNeuronIndex];
				changeSum /= Learn.batchSize;

				// Weights from source to target are stored in the source layer memory, target-neuron-wise.
				previousLayer.weights[neuronIndex * previousLayer.size + sourceNeuronIndex] += changeSum * LearningRate;
			}

			// Aggregate the bias changes by avergaging the sum value using the batch size and modify the model accordingly.
			neuron_bias changeSum = learningLayer.biasChanges[neuronIndex];
			changeSum /= Learn.batchSize;
			
			currentLayer.biases[neuronIndex] += changeSum * LearningRate;
		}
	}

	// All done !
}

float NN_Train_CPU(AIModel_NN& Model, const MNIST_Dataset& Dataset, size_t StartImageIndex, size_t EndImageIndex, float LearningRate)
{
	// Sanity checks
	if (	Model.layerCount < 3 || Model.layers == nullptr
		||	Dataset.imageCount < EndImageIndex
		)
		return -1.f;

	int iterationCount = EndImageIndex - StartImageIndex;
	if (iterationCount < 0)
	{
		iterationCount = Dataset.imageCount + iterationCount;
	}

	// In order to perform an epoch, Feedforward must be performed on the provided slice of the dataset using the Model.
	// At the end of each feedforward iteration, a "Learn Structure" is built by accumulation before it is passed to the final Backpropagation step.
	// Once backpropagation is done, the total output error is returned as indication of where the model was performance-wise *before* backpropagation happened.
	Feedforward_NN feedforward = InitFeedforwardInstance(Model);
	Learning_NN learn = InitLearningInstance(Model, iterationCount);

	float totalEpochCost = 0.f;
	int iterationIndex = 0;
	for (size_t imageIndex = StartImageIndex; imageIndex != EndImageIndex; imageIndex = ++imageIndex % Dataset.imageCount)
	{
		InitializeInputLayer(feedforward, Dataset.images[imageIndex]);
		PerformFeedforward(feedforward, Model);

		float iterationCost = PerformBackpropagation(Model, feedforward, Dataset.images[imageIndex], Dataset.labels[imageIndex], learn, iterationIndex++);
		if (iterationCost < 0.f)
		{
			// Something went wrong during backpropagation.
			return -1.f;
		}
#if 0
		printf("Results = [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\n",
			feedforward.layers[feedforward.layerCount - 1]->values[0], 
			feedforward.layers[feedforward.layerCount - 1]->values[1], 
			feedforward.layers[feedforward.layerCount - 1]->values[2], 
			feedforward.layers[feedforward.layerCount - 1]->values[3], 
			feedforward.layers[feedforward.layerCount - 1]->values[4],
			feedforward.layers[feedforward.layerCount - 1]->values[5], 
			feedforward.layers[feedforward.layerCount - 1]->values[6], 
			feedforward.layers[feedforward.layerCount - 1]->values[7], 
			feedforward.layers[feedforward.layerCount - 1]->values[8], 
			feedforward.layers[feedforward.layerCount - 1]->values[9]);
#endif

		totalEpochCost += iterationCost;
	}

	// Apply learning batch
	ApplyLearning(learn, Model, LearningRate);

	FreeFeedforwardInstance(feedforward);
	FreeLearningInstance(learn);

	return totalEpochCost / iterationCount;
}

