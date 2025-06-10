#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "cneuron/cneuron.h"

float sigmoid(float val, bool is_deravative);

dataset *get_xor();

dataset *get_or();

#endif
