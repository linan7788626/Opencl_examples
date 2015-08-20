#include <iostream>
#include <bitset>
#include <random>
#include <chrono>
#include <stdint.h>
#include <cassert>
#include <tuple>

#if 0
// C++11 random
std::random_device rd;
std::knuth_b gen(rd());

uint32_t genRandom()
{
    return gen();
}
#else
// bad, fast, random.

uint32_t genRandom()
{
    static uint32_t seed = std::random_device()();
    auto oldSeed = seed;
    seed = seed*1664525UL + 1013904223UL; // numerical recipes, 32 bit
    return oldSeed;
}
#endif

#ifdef _MSC_VER
uint32_t popcnt( uint32_t x ){ return _mm_popcnt_u32(x); }
#else
uint32_t popcnt( uint32_t x ){ return __builtin_popcount(x); }
#endif



std::pair<unsigned, unsigned> convolve()
{
    const uint32_t n = 6;
    const uint32_t iters = 1000;
    unsigned firstZero = 0;
    unsigned bothZero = 0;

    uint32_t S = (1 << (n+1));
    // generate all possible N+1 bit strings
    // 1 = +1
    // 0 = -1
    while ( S-- )
    {
        uint32_t s1 = S % ( 1 << n );
        uint32_t s2 = (S >> 1) % ( 1 << n );
        uint32_t fmask = (1 << n) -1; fmask |= fmask << 16;
        static_assert( n < 16, "packing of F fails when n > 16.");


        for( unsigned i = 0; i < iters; i++ )
        {
            // generate random bit mess
            uint32_t F;
            do {
                F = genRandom() & fmask;
            } while ( 0 == ((F % (1 << n)) ^ (F >> 16 )) );

            // Assume F is an array with interleaved elements such that F[0] || F[16] is one element
            // here MSB(F) & ~LSB(F) returns 1 for all elements that are positive
            // and  ~MSB(F) & LSB(F) returns 1 for all elements that are negative
            // this results in the distribution ( -1, 0, 0, 1 )
            // to ease calculations we generate r = LSB(F) and l = MSB(F)

            uint32_t r = F % ( 1 << n );
            // modulo is required because the behaviour of the leftmost bit is implementation defined
            uint32_t l = ( F >> 16 ) % ( 1 << n );

            uint32_t posBits = l & ~r;
            uint32_t negBits = ~l & r;
            assert( (posBits & negBits) == 0 );

            // calculate which bits in the expression S * F evaluate to +1
            unsigned firstPosBits = ((s1 & posBits) | (~s1 & negBits));
            // idem for -1
            unsigned firstNegBits = ((~s1 & posBits) | (s1 & negBits));

            if ( popcnt( firstPosBits ) == popcnt( firstNegBits ) )
            {
                firstZero++;

                unsigned secondPosBits = ((s2 & posBits) | (~s2 & negBits));
                unsigned secondNegBits = ((~s2 & posBits) | (s2 & negBits));

                if ( popcnt( secondPosBits ) == popcnt( secondNegBits ) )
                {
                    bothZero++;
                }
            }
        }
    }

    return std::make_pair(firstZero, bothZero);
}

int main()
{
    typedef std::chrono::high_resolution_clock clock;
    int rounds = 1000;
    std::vector< std::pair<unsigned, unsigned> > out(rounds);

    // do 100 rounds to get the cpu up to speed..
    for( int i = 0; i < 10000; i++ )
    {
        convolve();
    }


    auto start = clock::now();

    for( int i = 0; i < rounds; i++ )
    {
        out[i] = convolve();
    }

    auto end = clock::now();
    double seconds = std::chrono::duration_cast< std::chrono::microseconds >( end - start ).count() / 1000000.0;

#if 0
    for( auto pair : out )
        std::cout << pair.first << ", " << pair.second << std::endl;
#endif

    std::cout << seconds/rounds*1000 << " msec/round" << std::endl;

    return 0;
}
