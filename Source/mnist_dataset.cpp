#include "mnist_dataset.h"

#include <stdio.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

MNIST_Dataset ParseDataset(const uint8_t* LabelsBuffer, const uint8_t* ImagesBuffer)
{
	MNIST_Dataset dataset = {};
	size_t readOffset = 0;

	// Read header: Magic Number + label count.
	int32_t magicNumber_labels, labelCount;
	{
		magicNumber_labels = *(int32_t*)(LabelsBuffer + readOffset);
		readOffset += sizeof(uint32_t);
		EndianSwap(magicNumber_labels);

		labelCount = *(int32_t*)(LabelsBuffer + readOffset);
		readOffset += sizeof(uint32_t);
		EndianSwap(labelCount);
	}

	// Read in the labels
	dataset.labelCount = labelCount;
	dataset.labels = (int8_t*)malloc(labelCount);
	{
		if (dataset.labels == nullptr)
		{
			printf("Error: Out of memory for Training Labels Allocation.\n");
			return {};
		}

		memcpy(dataset.labels, LabelsBuffer + readOffset, labelCount);
	}

	// Read images.

	// Read header: Magic Number + Image Count + Row Count Per Image + Column Count Per Image
	readOffset = 0;
	int32_t magicNumber_img, imageCount, rowsPerImage, columnsPerImage;
	{
		magicNumber_img = *(int32_t*)(ImagesBuffer + readOffset);
		readOffset += sizeof(uint32_t);
		EndianSwap(magicNumber_img);

		imageCount = *(int32_t*)(ImagesBuffer + readOffset);
		readOffset += sizeof(uint32_t);
		EndianSwap(imageCount);

		rowsPerImage = *(int32_t*)(ImagesBuffer + readOffset);
		readOffset += sizeof(uint32_t);
		EndianSwap(rowsPerImage);

		columnsPerImage = *(int32_t*)(ImagesBuffer + readOffset);
		readOffset += sizeof(uint32_t);
		EndianSwap(columnsPerImage);
	}

	// Read in the images
	dataset.imageCount = imageCount;
	dataset.images = (MNIST_Img*)malloc(imageCount * sizeof(MNIST_Img));
	{
		if (dataset.images == nullptr)
		{
			printf("Error: Out of memory for Training Images allocation.\n");
			return {};
		}

		memcpy(dataset.images, ImagesBuffer + readOffset, sizeof(MNIST_Img) * imageCount); // The data should be organized row-wise already.
	}

	return dataset;
}

bool ExportMNISTImageAsBitmap(MNIST_Img& Image, const char* Filename)
{
	BITMAPFILEHEADER fileHeader = {};
	fileHeader.bfSize = sizeof(BITMAPFILEHEADER);
	fileHeader.bfType = 0x4d42;
	
	static BITMAPINFO* info = nullptr;
	static size_t infoSize = sizeof(BITMAPINFO) + sizeof(RGBQUAD) * 255;

	if (!info)
	{
		// Build info & color table

		info = (BITMAPINFO*)malloc(infoSize);
		if (!info)
		{
			printf("ERROR: Failed to allocate memory for Export Bitmap file info. Something is very wrong.\n");
			return false;
		}

		*info = {};

		info->bmiHeader.biSize = sizeof(BITMAPINFO);
		info->bmiHeader.biWidth = MNIST_Img::IMG_WIDTH;
		info->bmiHeader.biHeight = MNIST_Img::IMG_HEIGHT;
		info->bmiHeader.biPlanes = 1;
		info->bmiHeader.biBitCount = 8;
		info->bmiHeader.biCompression = BI_RGB;
		info->bmiHeader.biSizeImage = 0;
		info->bmiHeader.biClrUsed = 256;

		for (int colorIndex = 0; colorIndex < 256; colorIndex++)
		{
			info->bmiColors[colorIndex].rgbRed = colorIndex;
			info->bmiColors[colorIndex].rgbGreen = colorIndex;
			info->bmiColors[colorIndex].rgbBlue = colorIndex;
			info->bmiColors[colorIndex].rgbReserved = 0;
		}
	}

	fileHeader.bfSize += infoSize;

	// Build pixel buffer.
	uint8_t pixelBuffer[MNIST_Img::IMG_HEIGHT * MNIST_Img::IMG_WIDTH];
	for (int y = 0; y < MNIST_Img::IMG_HEIGHT; y++)
	{
		for (int x = 0; x < MNIST_Img::IMG_WIDTH; x++)
		{
			// Flip Y from in-memory image since images tend to be read bottom-up by image editors.
			pixelBuffer[y * MNIST_Img::IMG_WIDTH + x] = Image.pixelValues[((MNIST_Img::IMG_HEIGHT - 1) - y) * MNIST_Img::IMG_WIDTH + x];
		}
	}


	fileHeader.bfSize += sizeof(pixelBuffer);

	fileHeader.bfOffBits = sizeof(fileHeader) + infoSize;

	// Create file on platform and assemble it.
	HANDLE exportFile = CreateFile(Filename, GENERIC_WRITE, FILE_SHARE_WRITE, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	if (exportFile == INVALID_HANDLE_VALUE)
	{
		printf("ERROR: Failed to create export file '%s' ! Error Code = %d\n", Filename, GetLastError());
		return false;
	}

	uint8_t* finalBuffer = (uint8_t*)malloc(fileHeader.bfSize);
	if (!finalBuffer || fileHeader.bfSize < sizeof(fileHeader) + infoSize + sizeof(pixelBuffer))
	{
		printf("Failed to allocate memory for final file write, or inconsistent file header state. Aborting export.\n");
		return false;
	}

	memcpy_s(finalBuffer, fileHeader.bfSize, &fileHeader, sizeof(fileHeader));
	memcpy_s(finalBuffer + sizeof(fileHeader), fileHeader.bfSize - sizeof(fileHeader), info, infoSize);
	memcpy_s(finalBuffer + fileHeader.bfOffBits, fileHeader.bfSize - fileHeader.bfOffBits, pixelBuffer, sizeof(pixelBuffer));

	DWORD bytesWritten;
	WriteFile(exportFile, finalBuffer, fileHeader.bfSize, &bytesWritten, NULL);

	return true;
}