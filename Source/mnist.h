// Symbols for parsing and working with a MNIST dataset.

#ifndef MNIST_INCLUDED
#define MNIST_INCLUDED

#include <stdint.h>

#include <stdlib.h>

struct MNIST_Img
{
	static const size_t IMG_WIDTH = 28;
	static const size_t IMG_HEIGHT = 28;

	uint8_t pixelValues[28 * 28];
};

// Entire MNIST dataset. Since all such datasets in existence are relatively small, no need to add any streaming capabilities.
struct MNIST_Dataset
{
	int8_t* labels;
	int32_t labelCount;

	MNIST_Img* images;
	int32_t imageCount;
};

inline void EndianSwap(int32_t& num)
{
	num = ((num >> 24) & 0xff) | ((num >> 8) & 0xff00) | ((num << 8) & 0xff0000) | ((num << 24) & 0xff000000);
}

MNIST_Dataset ParseDataset(const uint8_t* LabelsBuffer, const uint8_t* ImagesBuffer);

// Exports a in-memory MNIST image as a simple 8bpp bitmap.
bool ExportMNISTImageAsBitmap(MNIST_Img& Image, const char* Filename);

#endif // MNIST_INCLUDED