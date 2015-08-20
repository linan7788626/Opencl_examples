std::pair<unsigned, unsigned> convolve()
{
    const uint32_t n = 6;
    const uint32_t iters = 1000;
    unsigned firstZero = 0;
    unsigned bothZero = 0;
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

        uint32_t mask = posBits | negBits;
        uint32_t totalBits = popcnt( mask );
        // if the amount of -1 and +1's is uneven, sum(S*F) cannot possibly evaluate to 0
        if ( totalBits & 1 )
            continue;

        uint32_t adjF = posBits & ~negBits;
        uint32_t desiredBits = totalBits / 2;

        uint32_t S = (1 << (n+1));
        // generate all possible N+1 bit strings
        // 1 = +1
        // 0 = -1
        while ( S-- )
        {
            // calculate which bits in the expression S * F evaluate to +1
            auto firstBits = (S & mask) ^ adjF;
            auto secondBits = (S & ( mask << 1 ) ) ^ ( adjF << 1 );

            bool a = desiredBits == popcnt( firstBits );
            bool b = desiredBits == popcnt( secondBits );
            firstZero += a;
            bothZero += a & b;
        }
    }

    return std::make_pair(firstZero, bothZero);
}
