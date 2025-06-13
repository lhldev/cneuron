#ifndef SHISHUA_RAND_H
#define SHISHUA_RAND_H

#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>

#include "shishua/shishua.h"

struct rand_chunk {
    size_t count;
    uint8_t buf[1024];  // NOTE: must be multiple of 256
};

uint8_t randnum_u8(uint8_t range, uint8_t offset);
uint32_t randnum_u32(uint32_t range, uint32_t offset);
float randf(float range, float offset);

extern struct rand_chunk randc;

extern prng_state __randstate;
#endif
