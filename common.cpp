#include <common.h>
#include <stdio.h>

void compare_arrays(int* a, int* b, int size) {
	for (int i = 0; i < size; i++) {
		if (a[i] != b[i]) {
			printf("arrays are different!");
			return;
		}
	}
	printf("Arrays are same!");
}