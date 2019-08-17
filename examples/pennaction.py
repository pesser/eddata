from eddata.pennaction import PennAction, PennActionCropped
from edflow.iterators.batches import plot_batch
import numpy as np

if __name__ == "__main__":
    d = PennAction()
    e = d[0]
    print(e)
    for k, v in e.items():
        print(k, type(v))

    print(sorted(set(d.labels["action"])))

    d = PennActionCropped()
    batch = np.stack([d[10 * i]["image"] for i in range(16)])
    plot_batch(batch, "pennaction_examples.png")
