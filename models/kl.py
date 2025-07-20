import numpy as np


def kl_bn(bn1,bn2):
    factors = [cpd.to_factor() for cpd in bn1.get_cpds()]
    joint1 = factors[0]
    for i in range(1, len(factors)):
        joint1 = joint1.product(factors[i], inplace=False)
    
    joint1.normalize()

    c1 = joint1.get_cardinality(joint1.scope())

    globalCardinality = np.prod(list(c1.values()))


    factors = [cpd.to_factor() for cpd in bn2.get_cpds()]

    joint2 = factors[0]

    for i in range(1,len(factors)):
        joint2 = joint2.product(factors[i], inplace=False)
    
    joint2.normalize()
    c2 = joint2.get_cardinality(joint2.scope())
    # print(np.prod(list(c2.values())))
    s = 0

    for i in range(0,globalCardinality):
        conf = joint1.assignment([i])[0]
        v1 = joint1.get_value(**dict(conf))
        v2 = joint2.get_value(**dict(conf))
        partial = v1 * (np.log(v1 + 1e-12) - np.log(v2 + 1e-12))
        s += partial

    return s

