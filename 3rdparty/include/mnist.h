#ifndef MNIST_H
#define MNIST_H
#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
typedef struct MNIST MNIST;
struct MNIST {
	float* image;
	int label;
};
inline void EndianChange(void *vp, size_t sz) {
	char *begin = (char *)vp;
	char *end = (char *)vp + sz - 1;
	while (begin < end) {
		*begin ^= *end;
		*end ^= *begin;
		*begin ^= *end;
		begin++;
		end--;
	}
}
inline MNIST* ReadMNIST(char* image_file, char* label_file, int* n, int padding) {
	int magic_number, num_of_images, rows, cols;
	FILE* fp_image = fopen(image_file, "rb");
	assert(fp_image != NULL);
	fread(&magic_number, sizeof(int), 1, fp_image); EndianChange(&magic_number, sizeof(int));
	assert(magic_number == 2051);
	fread(&num_of_images, sizeof(int), 1, fp_image); EndianChange(&num_of_images, sizeof(int));
	fread(&rows, sizeof(int), 1, fp_image); EndianChange(&rows, sizeof(int));
	fread(&cols, sizeof(int), 1, fp_image); EndianChange(&cols, sizeof(int));
	MNIST* ret = (MNIST*)calloc(num_of_images, sizeof(MNIST));
	FILE* fp_label = fopen(label_file, "rb");
	int num_of_items;
	assert(fp_label != NULL);
	fread(&magic_number, sizeof(int), 1, fp_label); EndianChange(&magic_number, sizeof(int));
	assert(magic_number == 2049);
	fread(&num_of_items, sizeof(int), 1, fp_label); EndianChange(&num_of_items, sizeof(int));
	assert(num_of_items == num_of_images);
	if (n != NULL) {
		*n = num_of_images;
	}
	unsigned char* p = (unsigned char*)calloc(rows*cols, sizeof(unsigned char));
	for (int n = 0; n<num_of_images; n++) {
		ret[n].image = (float*)calloc((rows + padding * 2)*(cols + padding * 2), sizeof(float));
		unsigned char label;
		fread(&label, sizeof(unsigned char), 1, fp_label);
		ret[n].label = label;
		fread(p, 1, rows*cols, fp_image);
		for (int y = 0; y<rows; y++) {
			for (int x = 0; x<cols; x++) {
				ret[n].image[(y + padding)*(cols + padding * 2) + (x + padding)] = (float)p[y*cols + x] / 255.0F;
			}
		}
	}
	free(p);
	return ret;
}
#endif
