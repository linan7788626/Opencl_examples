double f(double x)
{
    return x*x;
}

double Simple_Trap(double a, double b)
{
    double fA, fB;
    fA = f(a);
    fB = f(b);
    return ((fA + fB) * (b-a)) / 2;
}

double Comp_Trap( double a, double b)
{
    double Suma = 0;
    double i = 0;
    i = a + INC;
    Suma += Simple_Trap(a,i);
    while(i < b)
    {
        i+=INC;
        Suma += Simple_Trap(i,i + INC);
    }
    return Suma;
}
