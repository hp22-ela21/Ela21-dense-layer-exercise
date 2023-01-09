/********************************************************************************
* main.cpp: �vning g�llande implementering av ett densel-lager f�r enkelt 
*           neuralt n�tverk i C++. Komplettera medlemsfunktioner f�r strukten
*           dense_layer s� att insignaler { 1, 1, 1, 1 } medf�r predikterad 
*           utdata { 2, 2, 2 }, dvs. samma som tr�ningsdatan. 
********************************************************************************/
#include <vector>
#include "dense_layer.hpp"

/********************************************************************************
* main: Skapar ett dense-lager och tr�nar detta under tio epoker med en
*       l�rhastighet p� 10 %. Efter tr�ningen genomf�rs utskrift av 
*       dense-lagrets parametrar.
********************************************************************************/
int main(void)
{
   const std::vector<double> train_in = { 1, 1, 1, 1 };
   const std::vector<double> train_out = { 2, 2, 2 };

   dense_layer d1(3, 4);

   for (auto i = 0; i < 100; ++i)
   {
      d1.feedforward(train_in);
      d1.backpropagate(train_out);
      d1.optimize(train_in, 0.1);
   }

   d1.print();

   return 0;
}