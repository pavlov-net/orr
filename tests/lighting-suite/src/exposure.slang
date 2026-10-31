// Sets both exposure slots of a view to one pre-exposure, standing in for the output's meter.
#include "orr/root.h"
import view;

struct Root
{
    ViewConstants* view;
    float pre_exposure;
    uint _pad;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    for (uint slot = 0; slot < 2; slot++)
    {
        root.view.exposure[slot].pre_exposure = root.pre_exposure;
        root.view.exposure[slot].prev_pre_exposure = root.pre_exposure;
    }
}
