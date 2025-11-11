// Implementation of functions specific to the Neural Network AI model.

#include "model_nn.h"
#include "string.h"
#include "intrin.h"

#include <math.h>

#include <stdio.h>

#define INPUT_LAYER_SIZE (MNIST_Img::IMG_WIDTH * MNIST_Img::IMG_HEIGHT)
#define OUTPUT_LAYER_SIZE (AIModel_NN::OUTPUT_VALUE_COUNT)

#define RAND_BIAS_MIN ((neuron_bias)-0.2)
#define RAND_BIAS_MAX ((neuron_bias)0.2)

// Random weight value definitions. They need to be pretty low to avoid dealing with extremely large values during Feed Forward which can break the model.
#define RAND_WEIGHT_MIN ((neuron_weight)0)
#define RAND_WEIGHT_MAX ((neuron_weight)0.4)

neuron_bias GenRandomBias()
{
	if (RAND_BIAS_MAX - RAND_BIAS_MIN == 0) return 0;
	// Let's keep it really simple. Quantize whatever rand() returns as a thousandth and use that.
	int32_t randGen = rand() % (int32_t)((RAND_BIAS_MAX - RAND_BIAS_MIN) * 1000);
	return (randGen * (neuron_bias)0.001) + RAND_BIAS_MIN;
}

neuron_weight GenRandomWeight()
{
	// Let's keep it really simple. Quantize whatever rand() returns as a thousandth and use that.
	int32_t randGen = rand() % (int32_t)((RAND_WEIGHT_MAX - RAND_WEIGHT_MIN) * 1000);
	return (randGen * (neuron_weight)0.001) + RAND_WEIGHT_MIN;
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

void NN_FreeModel(AIModel_NN& Model)
{
	// The Model's entire memory is tied to a single allocation at the location of its layers pointer.
	free(Model.layers);
}

neuron_activation Activation(neuron_activation inActivation)
{
	return inActivation < 0 ? inActivation * 0.1f : inActivation;
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

// Determines the Output Error of the given Feedforward result against the input Image and Label, and performs backpropagation according to it.
// Returns the total Error "processed" for this iteration.
float PerformBackpropagation(AIModel_NN& Model, const Feedforward_NN& Feedforward, const MNIST_Img& InputImage, const int8_t& InputLabel, size_t IterationIndex)
{
	static float constexpr LEARNING_RATE = 0.0001f;

	// Check that label value is valid.
	if (InputLabel > 9) return -1.f;

	// Use a double-buffered system for holding Desired Values for the current and previous layer while backpropagating.
	// They are as large as the largest layer in the model, meaning no bound checking or extra allocations are ever required for them in the whole backpropagation process.
	neuron_activation* currentDesiredValues;
	neuron_activation* previousLayerDesiredValues;
	size_t desiredValuesBufferSize;
	{
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
		previousLayerDesiredValues = (neuron_activation*)malloc(desiredValuesBufferSize);

		if (currentDesiredValues == nullptr || previousLayerDesiredValues == nullptr)
		{
			return -1.f;
		}

		memset(currentDesiredValues, 0, desiredValuesBufferSize);
		memset(previousLayerDesiredValues, 0, desiredValuesBufferSize);
	}

	// Initialize the output layer desired values according to the label.

	currentDesiredValues[InputLabel] = 1;

	float totalCost = 0.f;

	// Work our way backwards starting from the output layer.
	for (int layerIndex = Model.layerCount - 1; layerIndex > 0; layerIndex--)
	{
		// For each neuron in the layer, determine the error between the corresponding FeedforwardResult value and desired value.
		// Then, determine a change in inbound weight values and bias to reduce this error.
		// Once the weights and the bias are updated, determine the relative error for all inbound neurons and add it to their place in the previous layer desired values.
		// Those sums will then be converted back to an actual desired value as a post processing step using their actual value during the feed forward phase, saving memory.
		for (int neuronIndex = 0; neuronIndex < Model.layers[layerIndex]->size; neuronIndex++)
		{
			neuron_activation currentValue = Feedforward.layers[layerIndex]->values[neuronIndex];
			neuron_activation currentDesired = currentDesiredValues[neuronIndex];
			neuron_activation relativeError = currentValue - currentDesired;
			
			// To bring all errors into the positive space (so they don't cancel one another out), square them.
			neuron_activation cost = relativeError * relativeError;

			// Add to the total error if on output layer.
			if (layerIndex == Model.layerCount - 1)
			totalCost += cost;

			// Begin gradient descent by adjusting each inbound weight depending on their importance (which depends solely on the activation value of the associated previous layer
			// neuron). At the same time, determine a desired value for that neuron.
			if (cost > 0.0005f)
			{
				for (int previousLayerNeuronIndex = 0; previousLayerNeuronIndex < Model.layers[layerIndex - 1]->size; previousLayerNeuronIndex++)
				{
					neuron_weight& inboundWeight = Model.layers[layerIndex - 1]->weights[neuronIndex * Model.layers[layerIndex - 1]->size + previousLayerNeuronIndex];
					const neuron_activation& inboundValue = Feedforward.layers[layerIndex - 1]->values[previousLayerNeuronIndex];

					// Determine gradient for inbound weight.
					float costDeltaByWeight = (currentValue < 0 ? inboundValue * 0.1f : inboundValue) * 2 * (currentValue - currentDesired);
					inboundWeight += LEARNING_RATE * -costDeltaByWeight;

					// Determine gradient for previous layer neuron.
					float costDeltaByInboundValue = (currentValue < 0 ? inboundWeight * 0.1f: inboundWeight) * 2 * (currentValue - currentDesired);
					previousLayerDesiredValues[previousLayerNeuronIndex] += LEARNING_RATE * -costDeltaByInboundValue;
				}

				neuron_bias& currentBias = Model.layers[layerIndex]->biases[neuronIndex];

				float costDeltaByBias = 2 * (currentValue - currentDesired);
				currentBias += LEARNING_RATE * -costDeltaByBias;
			}
		}

		// Post Process the Previous Layer Desired Values buffer - it for now contains the sum of the relative errors. Switch it back to desired values by offsetting it
		// by the corresponding neuron activation values in that layer.
		if (layerIndex - 1 > 0)
		for (int previousLayerNeuronIndex = 0; previousLayerNeuronIndex < Model.layers[layerIndex - 1]->size; previousLayerNeuronIndex++)
		{
			previousLayerDesiredValues[previousLayerNeuronIndex] /= Model.layers[layerIndex]->size;
			previousLayerDesiredValues[previousLayerNeuronIndex] += Feedforward.layers[layerIndex - 1]->values[previousLayerNeuronIndex];
			previousLayerDesiredValues[previousLayerNeuronIndex] = Activation(previousLayerDesiredValues[previousLayerNeuronIndex]);
		}

		// Switch the desired value buffers.
		neuron_activation* swap = currentDesiredValues;
		currentDesiredValues = previousLayerDesiredValues;
		previousLayerDesiredValues = swap;

		// Zero out previous layer buffer to make sure no unrelated information survives to the next iteration.
		memset(previousLayerDesiredValues, 0, desiredValuesBufferSize);
	}

	// Free allocated desired value buffers.
	free(currentDesiredValues);
	free(previousLayerDesiredValues);

	return totalCost;
}

float NN_Train_CPU(AIModel_NN& Model, const MNIST_Dataset& Dataset, size_t StartImageIndex, size_t EndImageIndex)
{
	// Sanity checks
	if (	Model.layerCount < 3 || Model.layers == nullptr
		||	Dataset.imageCount < EndImageIndex
		)
		return -1.f;

	// In order to perform an epoch, Feedforward must be performed on the provided slice of the dataset using the Model.
	// At the end of each feedforward iteration, a "Learn Structure" is built by accumulation before it is passed to the final Backpropagation step.
	// Once backpropagation is done, the total output error is returned as indication of where the model was performance-wise *before* backpropagation happened.
	Feedforward_NN feedforward = InitFeedforwardInstance(Model);
	float totalEpochError = 0.f;
	int iterationCount = 0;
	for (size_t imageIndex = StartImageIndex; imageIndex != EndImageIndex; imageIndex = ++imageIndex % Dataset.imageCount)
	{
		InitializeInputLayer(feedforward, Dataset.images[imageIndex]);
		PerformFeedforward(feedforward, Model);

		float iterationError = PerformBackpropagation(Model, feedforward, Dataset.images[imageIndex], Dataset.labels[imageIndex], iterationCount);
		if (iterationError < 0.f)
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

		totalEpochError += iterationError;
		iterationCount++;
	}

	FreeFeedforwardInstance(feedforward);

	return totalEpochError / iterationCount;
}

