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
				newModel.layers[layerIndex]->biases[neuronIndex] = GenRandomBias(0.01, 0.1f);
			}
		}
		if (bRandomWeights && newModel.layers[layerIndex]->weights != nullptr) // Obviously only randomize the weights up until the second-to-last layer.
			// I'd rather have checked the layer index but Intellisense was "warning" me that dereferencing ->weights was dangerous unless I explicitly checked the pointer's
			// non-nullity. Thanks Microsoft.
		{
			for (uint16_t targetNeuronIndex = 0; targetNeuronIndex < newModel.layers[layerIndex + 1]->size; targetNeuronIndex++)
			{
				for (uint16_t neuronIndex = 0; neuronIndex < newModel.layers[layerIndex]->size; neuronIndex++)
				{
					newModel.layers[layerIndex]->weights[targetNeuronIndex * newModel.layers[layerIndex]->size + neuronIndex] = GenRandomWeight(-0.1, 0.1);
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
#if 0
	return (Activation - Desired) * (Activation - Desired);
#else
	return (Desired * log(Activation));
#endif
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
}

// Applies the Softmax function onto the input values and stores the resulting values into the outputValues buffer.
void Softmax(neuron_activation* inputValues, size_t inputValueCount, neuron_activation* outputValues)
{
	// Find maximum for first normalization pass.
	// It turns out this is necessary to prevent very high output values from breaking the algorithm.
	neuron_activation maxOutput = inputValues[0];
	for (int outputNeuronIndex = 1; outputNeuronIndex < OUTPUT_LAYER_SIZE; outputNeuronIndex++)
	{
		neuron_activation output = inputValues[outputNeuronIndex];
		if (output > maxOutput)
		{
			maxOutput = output;
		}
	}

	if (maxOutput * maxOutput < 1)
	{
		maxOutput = 1; // No normalization for small max absolute output values.
	}

	// Compute transformed values and their sum.
	double_t transformedSum = 0;
	for (int outputNeuronIndex = 0; outputNeuronIndex < OUTPUT_LAYER_SIZE; outputNeuronIndex++)
	{
		// Raise exponential to the value of the output normalized by maximum for greater number (and system) stability.
		outputValues[outputNeuronIndex] = exp(inputValues[outputNeuronIndex] - maxOutput);
		transformedSum += outputValues[outputNeuronIndex];
	}

	// Compute normalized transformed values.
	for (int outputNeuronIndex = 0; outputNeuronIndex < OUTPUT_LAYER_SIZE; outputNeuronIndex++)
	{
		outputValues[outputNeuronIndex] /= transformedSum;
	}
}

// Read the output values of a feedforward structure and convert them to a valid probability distribution within a Result structure.
FeedforwardResult_NN ExtractResults(const Feedforward_NN& Feedforward)
{
	FeedforwardResult_NN Result = {};

	// Calculate Softmax of output layer, to be used as the "actual" output of the network.
	neuron_activation networkOutputs[OUTPUT_LAYER_SIZE];
	Softmax(Feedforward.layers[Feedforward.layerCount - 1]->values, OUTPUT_LAYER_SIZE, networkOutputs);

	// Copy softmaxed values to feedforward result values.
	memcpy(&Result.values, networkOutputs, OUTPUT_LAYER_SIZE * sizeof(neuron_activation));

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
float PerformBackpropagation(const AIModel_NN& Model, const Feedforward_NN& Feedforward, const MNIST_Img& InputImage, int8_t InputLabel, Learning_NN& Learning, size_t IterationIndex)
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
	neuron_activation* currentLayerOutputDelta = nullptr;
	neuron_activation* sourceLayerOutputDelta = nullptr;
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
	currentLayerOutputDelta = (neuron_activation*)malloc(desiredValuesBufferSize);
	sourceLayerOutputDelta = (neuron_activation*)malloc(desiredValuesBufferSize);

	if (currentLayerOutputDelta == nullptr || sourceLayerOutputDelta == nullptr)
	{
		return -1.f;
	}

	memset(currentLayerOutputDelta, 0, desiredValuesBufferSize);
	memset(sourceLayerOutputDelta, 0, desiredValuesBufferSize);

	// Calculate Softmax of output layer, to be used as the "actual" output of the network for the purpose of initial error calculation.
	neuron_activation networkOutputs[OUTPUT_LAYER_SIZE];
	Softmax(Feedforward.layers[Feedforward.layerCount - 1]->values, OUTPUT_LAYER_SIZE, networkOutputs);

	// Calculate total output cost. The goal of Backpropagation is to lower this value.
	double totalOutputLoss = 0.f;

	// Add to the total error if on output layer.
	for (int outputNeuronIndex = 0; outputNeuronIndex < OUTPUT_LAYER_SIZE; outputNeuronIndex++)
	{
		neuron_activation output = networkOutputs[outputNeuronIndex];
		double cost = Cost(output, outputNeuronIndex == InputLabel);
		totalOutputLoss -= cost;
	}

	// Backpropagation
	// Work our way backwards starting from the output layer.

	// OUTPUT LAYER -> LAST HIDDEN LAYER
	// The output layer gets Softmax applied to its values so its gradients are different.
	{
		int layerIndex = Model.layerCount - 1;
		// Learning Layer in which to record the desired changes in weights and biases.
		Learning_NN::Layer& learningLayer = *Learning.layers[layerIndex - 1];

		// For each neuron in the output, determine the error between the corresponding FeedforwardResult value and desired value.
		// Then, determine a change in inbound weight values and bias to reduce this error.
		// Determine the relative error for all inbound neurons and add it to their place in the previous layer desired values array.
		// Those sums will then be converted back to an actual desired value as a post processing step using their actual value during the feed forward phase, saving memory.
		for (int neuronIndex = 0; neuronIndex < OUTPUT_LAYER_SIZE; neuronIndex++)
		{
			const neuron_activation& outputActivation = Feedforward.layers[layerIndex]->values[neuronIndex]; // Pre-softmax output value
			const neuron_activation& outputValue = networkOutputs[neuronIndex]; // Post-softmax output value
			neuron_activation outputCost = -Cost(outputValue, neuronIndex == InputLabel);

			// Compute output delta, the change in overall error for a change in the pre-softmax value of this neuron.
			neuron_activation outputDelta = 0;
			//if (neuronIndex == InputLabel)
			//{
			//	outputDelta = -1 / (outputValue * (1 - outputValue));
			//}
			//else
			//{
			//	outputDelta = -1 / (-outputValue * networkOutputs[InputLabel]);
			//}

			if (neuronIndex == InputLabel) {
				outputDelta = (outputValue - 1.0) * (1.0 - outputValue);
			}
			else {
				outputDelta = (outputValue) * (1.0 - outputValue);
			}

			// Bias gradient descent
			// Determine the change in value for all output layer neurons and average them to obtain the "global" descent for the bias.
			neuron_activation biasGradient = d_Activation(outputActivation); // Change in pre-softmax value of the output for each added bias.
			learningLayer.biasChanges[neuronIndex] += -(biasGradient * outputDelta);

			// Weight gradient descent
			// For each source neuron, determine the change in value for all output layer neurons and average them to obtain the "global" descent for the source weight.
			for (int sourceNeuronIndex = 0; sourceNeuronIndex < Model.layers[layerIndex - 1]->size; sourceNeuronIndex++)
			{
				const neuron_activation& sourceActivation = Feedforward.layers[layerIndex - 1]->values[sourceNeuronIndex];
				
				neuron_activation weightGradient = sourceActivation * d_Activation(outputActivation); // Change in pre-softmax value of this neuron for a change in weight of source neuron.
				learningLayer.weightChanges[neuronIndex * Model.layers[layerIndex - 1]->size + sourceNeuronIndex] -= weightGradient* outputDelta;
			}

			// Determine the error to add to each source neuron.
			// To accomplish this, use the neuron's pre-softmax output value so it speaks the same "language" as the hidden layers in terms of scale,
			// by setting the source neuron's output delta to this neuron's output delta weighed by... weight.

			for (int sourceNeuronIndex = 0; sourceNeuronIndex < Model.layers[layerIndex - 1]->size; sourceNeuronIndex++)
			{ 
				const neuron_weight& sourceWeight = Model.layers[layerIndex - 1]->weights[neuronIndex * Model.layers[layerIndex - 1]->size + sourceNeuronIndex];
				sourceLayerOutputDelta[sourceNeuronIndex] += outputDelta * sourceWeight;
			}
		}

		// Switch the output delta buffers.
		neuron_activation* swap = currentLayerOutputDelta;
		currentLayerOutputDelta = sourceLayerOutputDelta;
		sourceLayerOutputDelta = swap;

		// Zero out source layer buffer to make sure no unrelated information survives to the next iteration.
		memset(sourceLayerOutputDelta, 0, desiredValuesBufferSize);
	}
	
	// HIDDEN LAYERS -> INPUT LAYER
	for (int layerIndex = Model.layerCount - 2; layerIndex > 0; layerIndex--)
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
			neuron_activation outputDelta = currentLayerOutputDelta[neuronIndex];

			// Bias gradient descent

			neuron_activation biasGradient = d_Activation(outputActivation); // Change in pre-softmax value of the output for each added bias.
			learningLayer.biasChanges[neuronIndex] += -(biasGradient * outputDelta);

			for (int sourceNeuronIndex = 0; sourceNeuronIndex < Model.layers[layerIndex - 1]->size; sourceNeuronIndex++)
			{
				// Weight gradient descent
				neuron_activation sourceActivation = Feedforward.layers[layerIndex - 1]->values[sourceNeuronIndex];
				neuron_weight weightGradient = sourceActivation * d_Activation(outputActivation);

				learningLayer.weightChanges[neuronIndex * Model.layers[layerIndex - 1]->size + sourceNeuronIndex] -= weightGradient * outputDelta;
			}

			// Sum up output deltas for previous layer.
			if (layerIndex > 1)
			for (int sourceNeuronIndex = 0; sourceNeuronIndex < Model.layers[layerIndex - 1]->size; sourceNeuronIndex++)
			{
				const neuron_weight& sourceWeight = Model.layers[layerIndex - 1]->weights[neuronIndex * Model.layers[layerIndex - 1]->size + sourceNeuronIndex];
				sourceLayerOutputDelta[sourceNeuronIndex] += outputDelta * sourceWeight;
			}
		}

		// Switch the desired value buffers.
		neuron_activation* swap = currentLayerOutputDelta;
		currentLayerOutputDelta = sourceLayerOutputDelta;
		sourceLayerOutputDelta = swap;

		// Zero out previous layer buffer to make sure no unrelated information survives to the next iteration.
		memset(sourceLayerOutputDelta, 0, desiredValuesBufferSize);
	}

	// Free allocated desired value buffers.
	free(currentLayerOutputDelta);
	free(sourceLayerOutputDelta);

	return totalOutputLoss;
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

#if 0
	for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
	{
		printf("Bias of output %d = %f\n", i, Model.layers[Model.layerCount - 1]->biases[i]);
	}
#endif

	FreeFeedforwardInstance(feedforward);
	FreeLearningInstance(learn);

	return totalEpochCost / iterationCount;
}

