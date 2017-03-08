// CS 61C Fall 2015 Project 4

// include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

// include OpenMP
#if !defined(_MSC_VER)
#include <pthread.h>
#endif
#include <omp.h>

#include "calcDepthOptimized.h"
#include "calcDepthNaive.h"

/* DO NOT CHANGE ANYTHING ABOVE THIS LINE. */


void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement) {

	memset(depth, 0, imageHeight * imageWidth * sizeof(float));
	int unroll8 = (2 * featureWidth + 1) / 8 * 8;
	int unroll4 = (2 * featureWidth + 1) / 4 * 4;
	int unroll1 = (2 * featureWidth + 1);

#pragma omp parallel
	{
#pragma omp for
		for (int y = featureHeight; y < imageHeight - featureHeight; y++) {
			for (int x = featureWidth; x < imageWidth - featureWidth; x++) {

				float minimumSquaredDifference = -1;
				int minimumDy = 0;
				int minimumDx = 0;

				for (int dy = -maximumDisplacement; dy <= maximumDisplacement; dy++) {

					for (int dx = -maximumDisplacement; dx <= maximumDisplacement; dx++) {

						if (y + dy - featureHeight < 0 || y + dy + featureHeight >= imageHeight ||
							x + dx - featureWidth < 0 || x + dx + featureWidth >= imageWidth) {
							continue;
						}

						float squaredDifference = 0;
						float sumArr[4];
						__m128 tempSum = _mm_setzero_ps();


						for (int boxX = 0; boxX < unroll8; boxX += 8) {

							int leftX = x + boxX;
							int rightX = x + dx + boxX;

							for (int boxY = -featureHeight; boxY <= featureHeight; boxY++) {
								int leftY = (y + boxY) * imageWidth + leftX - featureWidth;
								int rightY = (y + boxY + dy) * imageWidth + rightX - featureWidth;

								__m128 leftVal = _mm_loadu_ps(left + leftY);
								__m128 rightVal = _mm_loadu_ps(right + rightY);
								__m128 diff = _mm_sub_ps(leftVal, rightVal);
								__m128 mult = _mm_mul_ps(diff, diff);
								tempSum = _mm_add_ps(mult, tempSum);

								leftVal = _mm_loadu_ps(left + leftY + 4);
								rightVal = _mm_loadu_ps(right + rightY + 4);
								diff = _mm_sub_ps(leftVal, rightVal);
								mult = _mm_mul_ps(diff, diff);
								tempSum = _mm_add_ps(mult, tempSum);
							}

						}

						for (int boxX = unroll8; boxX < unroll4; boxX += 4) {

							int leftX = x + boxX;
							int rightX = x + dx + boxX;

							for (int boxY = -featureHeight; boxY <= featureHeight; boxY++) {
								int leftY = (y + boxY) * imageWidth + leftX - featureWidth;
								int rightY = (y + boxY + dy) * imageWidth + rightX - featureWidth;

								__m128 leftVal = _mm_loadu_ps(left + leftY);
								__m128 rightVal = _mm_loadu_ps(right + rightY);
								__m128 diff = _mm_sub_ps(leftVal, rightVal);
								__m128 mult = _mm_mul_ps(diff, diff);
								tempSum = _mm_add_ps(mult, tempSum);
							}

						}

						for (int boxX = unroll4; boxX < unroll1; boxX += 1) {

							int leftX = x + boxX - featureWidth;
							int rightX = x + dx + boxX - featureWidth;

							for (int boxY = -featureHeight; boxY <= featureHeight; boxY++) {
								int leftY = y + boxY;
								int rightY = y + dy + boxY;

								float difference = left[leftY * imageWidth + leftX] - right[rightY * imageWidth + rightX];
								squaredDifference += difference * difference;
							}

						}

						_mm_storeu_ps(sumArr, tempSum);
						squaredDifference += sumArr[0] + sumArr[1] + sumArr[2] + sumArr[3];


						if ((minimumSquaredDifference == -1) || ((minimumSquaredDifference == squaredDifference) &&
																 (displacementNaive(dx, dy) <
																  displacementNaive(minimumDx, minimumDy))) ||
							(minimumSquaredDifference > squaredDifference)) {
							minimumSquaredDifference = squaredDifference;
							minimumDx = dx;
							minimumDy = dy;
						}
					}

				}


				if (minimumSquaredDifference != -1) {
					if (maximumDisplacement == 0) {
						depth[y * imageWidth + x] = 0;
					} else {
						depth[y * imageWidth + x] = displacementNaive(minimumDx, minimumDy);
					}
				} else {
					depth[y * imageWidth + x] = 0;
				}
			}
		}
	}

}

