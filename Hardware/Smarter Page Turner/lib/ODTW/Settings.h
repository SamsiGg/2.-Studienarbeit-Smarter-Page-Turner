#ifndef SETTINGS_H
#define SETTINGS_H

#define FFT_SIZE 4096
#define NUM_CHROMA 12

// DTW Parameter
#define PENALTY_WAIT 2.0f
#define PENALTY_STEP 0.0f
#define PENALTY_SKIP 0.8f

// Optimierung: Radius verkleinern
#define CALC_RADIUS 100     // +/- 100 Frames reichen meistens

#define PAGE_TURN_OFFSET 10 
#define START_THRESHOLD 1500 

#endif