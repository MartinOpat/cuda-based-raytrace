#ifndef CONSTS_H
#define CONSTS_H

// TODO: Eventually, export this into a better place (i.e., such that we do not have to recompile every time we change a parameter)
const int VOLUME_WIDTH  = 49;
const int VOLUME_HEIGHT = 51;
const int VOLUME_DEPTH  = 42;

const int IMAGE_WIDTH   = 2560;
const int IMAGE_HEIGHT  = 1440;

const int SAMPLES_PER_PIXEL = 8;  // TODO: Right now uses simple variance, consider using something more advanced (e.g., some commonly-used noise map)

#endif // CONSTS_H