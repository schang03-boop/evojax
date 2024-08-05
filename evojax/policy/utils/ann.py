import jax.numpy as jnp
import jax


# -- ANN Ordering -------------------------------------------------------- -- #

def get_node_order(nodeG, connG):
    """Builds connection matrix from genome through topological sorting.

    Args:
      nodeG - (jnp_array) - node genes
              [3 X nUniqueGenes]
              [0,:] == Node Id
              [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
              [2,:] == Activation function (as int)

      connG - (jnp_array) - connection genes
              [5 X nUniqueGenes]
              [0,:] == Innovation Number (unique Id)
              [1,:] == Source Node Id
              [2,:] == Destination Node Id
              [3,:] == Weight Value
              [4,:] == Enabled?

    Returns:
      Q    - [int]      - sorted node order as indices
      wMat - (jnp_array) - ordered weight matrix
             [N X N]

      OR

      False, False      - if cycle is found
    """
    conn = jnp.copy(connG)
    node = jnp.copy(nodeG)
    nIns = len(node[0, node[1, :] == 1]) + len(node[0, node[1, :] == 4])
    nOuts = len(node[0, node[1, :] == 2])

    # Create connection and initial weight matrices
    conn = conn.at[3, conn[4, :] == 0].set(jnp.nan)  # disabled but still connected
    src = conn[1, :].astype(int)
    dest = conn[2, :].astype(int)

    lookup = node[0, :].astype(int)
    for i in range(len(lookup)):  # Can we vectorize this?
        src = src.at[jnp.where(src == lookup[i])].set(i)
        dest = dest.at[jnp.where(dest == lookup[i])].set(i)

    wMat = jnp.zeros((jnp.shape(node)[1], jnp.shape(node)[1]))
    wMat = wMat.at[src, dest].set(conn[3, :])
    connMat = wMat[nIns + nOuts:, nIns + nOuts:]
    connMat = connMat.at[connMat != 0].set(1)

    # Topological Sort of Hidden Nodes
    edge_in = jnp.sum(connMat, axis=0)
    Q = jnp.where(edge_in == 0)[0]  # Start with nodes with no incoming connections
    for i in range(len(connMat)):
        if (len(Q) == 0) or (i >= len(Q)):
            Q = jnp.array([])
            return False, False  # Cycle found, can't sort
        edge_out = connMat[Q[i], :]
        edge_in = edge_in - edge_out  # Remove nodes' conns from total
        nextNodes = jnp.setdiff1d(jnp.where(edge_in == 0)[0], Q)
        Q = jnp.hstack((Q, nextNodes))

        if jnp.sum(edge_in) == 0:
            break

    # Add In and outs back and reorder wMat according to sort
    Q = Q + nIns + nOuts
    Q = jnp.r_[lookup[:nIns], Q, lookup[nIns:nIns + nOuts]]
    wMat = wMat[jnp.ix_(Q, Q)]

    return Q, wMat


def getLayer(wMat):
    """Get layer of each node in weight matrix
    Traverse wMat by row, collecting layer of all nodes that connect to you (X).
    Your layer is max(X)+1. Input and output nodes are ignored and assigned layer
    0 and max(X)+1 at the end.

    Args:
      wMat  - (jnp_array) - ordered weight matrix
             [N X N]

    Returns:
      layer - [int]      - layer # of each node

    Todo:
      * With very large networks this might be a performance sink -- especially,
      given that this happen in the serial part of the algorithm. There is
      probably a more clever way to do this given the adjacency matrix.
    """
    wMat = wMat.at[jnp.isnan(wMat)].set(0)
    wMat = wMat.at[wMat != 0].set(1)
    nNode = jnp.shape(wMat)[0]
    layer = jnp.zeros((nNode,))
    while True:  # Loop until sorting is stable
        prevOrder = jnp.copy(layer)
        for curr in range(nNode):
            srcLayer = jnp.zeros((nNode,))
            for src in range(nNode):
                srcLayer = srcLayer.at[src].set(layer[src] * wMat[src, curr])
            layer = layer.at[curr].set(jnp.max(srcLayer) + 1)
        if jnp.all(prevOrder == layer):
            break
    return layer - 1


# -- ANN Activation ------------------------------------------------------ -- #

def act(weights, aVec, nInput, nOutput, inPattern):
    """Returns FFANN output given a single input pattern
    If the variable weights is a vector it is turned into a square weight matrix.

    Allows the network to return the result of several samples at once if given a matrix instead of a vector of inputs:
        Dim 0 : individual samples
        Dim 1 : dimensionality of pattern (# of inputs)

    Args:
      weights   - (jnp_array) - ordered weight matrix or vector
                  [N X N] or [N**2]
      aVec      - (jnp_array) - activation function of each node
                  [N X 1]    - stored as ints (see applyAct in ann.py)
      nInput    - (int)      - number of input nodes
      nOutput   - (int)      - number of output nodes
      inPattern - (jnp_array) - input activation
                  [1 X nInput] or [nSamples X nInput]

    Returns:
      output    - (jnp_array) - output activation
                  [1 X nOutput] or [nSamples X nOutput]
    """
    # Turn weight vector into weight matrix
    if jnp.ndim(weights) < 2:
        nNodes = int(jnp.sqrt(jnp.shape(weights)[0]))
        wMat = jnp.reshape(weights, (nNodes, nNodes))
    else:
        nNodes = jnp.shape(weights)[0]
        wMat = weights
    wMat = wMat.at[jnp.isnan(wMat)].set(0)

    # Vectorize input
    if jnp.ndim(inPattern) > 1:
        nSamples = jnp.shape(inPattern)[0]
    else:
        nSamples = 1

    # Run input pattern through ANN
    nodeAct = jnp.zeros((nSamples, nNodes))
    nodeAct = nodeAct.at[:, 0].set(1)  # Bias activation
    nodeAct = nodeAct.at[:, 1:nInput + 1].set(inPattern)

    # Propagate signal through hidden to output nodes
    iNode = nInput + 1
    for iNode in range(nInput + 1, nNodes):
        rawAct = jnp.dot(nodeAct, wMat[:, iNode]).squeeze()
        nodeAct = nodeAct.at[:, iNode].set(applyAct(aVec[iNode], rawAct))
        # print(nodeAct)
    output = nodeAct[:, -nOutput:]
    return output


def applyAct(actId, x):
    """Returns value after an activation function is applied
    Lookup table to allow activations to be stored in jnp arrays

    case 1  -- Linear
    case 2  -- Unsigned Step Function
    case 3  -- Sin
    case 4  -- Gaussian with mean 0 and sigma 1
    case 5  -- Hyperbolic Tangent [tanh] (signed)
    case 6  -- Sigmoid unsigned [1 / (1 + exp(-x))]
    case 7  -- Inverse
    case 8  -- Absolute Value
    case 9  -- Relu
    case 10 -- Cosine
    case 11 -- Squared

    Args:
      actId   - (int)   - key to look up table
      x       - (???)   - value to be input into activation
                [? X ?] - any type or dimensionality

    Returns:
      output  - (float) - value after activation is applied
                [? X ?] - same dimensionality as input
    """
    if actId == 1:  # Linear
        value = x

    if actId == 2:  # Unsigned Step Function
        value = 1.0 * (x > 0.0)
        # value = (jnp.tanh(50*x/2.0) + 1.0)/2.0

    elif actId == 3:  # Sin
        value = jnp.sin(jnp.pi * x)

    elif actId == 4:  # Gaussian with mean 0 and sigma 1
        value = jnp.exp(-jnp.multiply(x, x) / 2.0)

    elif actId == 5:  # Hyperbolic Tangent (signed)
        value = jnp.tanh(x)

    elif actId == 6:  # Sigmoid (unsigned)
        value = (jnp.tanh(x / 2.0) + 1.0) / 2.0

    elif actId == 7:  # Inverse
        value = -x

    elif actId == 8:  # Absolute Value
        value = jnp.abs(x)

    elif actId == 9:  # Relu
        value = jnp.maximum(0, x)

    elif actId == 10:  # Cosine
        value = jnp.cos(jnp.pi * x)

    elif actId == 11:  # Squared
        value = x ** 2

    else:
        value = x
    return value


# -- Action Selection ---------------------------------------------------- -- #

def selectAct(action, actSelect):
    """Selects action based on vector of actions

      Single Action:
      - Hard: a single action is chosen based on the highest index
      - Prob: a single action is chosen probabilistically with higher values
              more likely to be chosen


      We aren't selecting a single action:
      - Softmax: a softmax normalized distribution of values is returned
      - Default: all actions are returned


    Args:
      action   - (jnp_array) - vector weighting each possible action
                  [N X 1]


    Returns:
      i         - (int) or (jnp_array)     - chosen index
                           [N X 1]

    """
    if actSelect == 'softmax':
        action = softmax(action)

    elif actSelect == 'prob':
        action = weightedRandom(jnp.sum(action, axis=0))

    else:
        action = action.flatten()

    return action


def softmax(x):
    """Compute softmax values for each sets of scores in x.
    Assumes: [samples x dims]

    Args:
      x - (jnp_array) - normalized values
          [samples x dims]

    Returns:
      softmax - (jnp_array) - softmax normalized in dim 1
    Todo: Untangle all the transposes...
    """
    if x.ndim == 1:
        e_x = jnp.exp(x - jnp.max(x))
        return e_x / e_x.sum(axis=0)

    else:
        e_x = jnp.exp(x.T - jnp.max(x, axis=1))
        return (e_x / e_x.sum(axis=0)).T


def weightedRandom(weights):
    """Returns random index, with each choices chance weighted

    Args:
      weights   - (jnp_array) - weighting of each choice
                  [N X 1]


    Returns:
      i         - (int)      - chosen index
    """
    minVal = jnp.min(weights)
    weights = weights - minVal  # handle negative vals
    cumVal = jnp.cumsum(weights)
    pick = jax.random.uniform(jax.random.PRNGKey(0), (), minval=0, maxval=cumVal[-1])

    for i in range(len(weights)):
        if cumVal[i] >= pick:
            return i

# -- File I/O ------------------------------------------------------------ -- #

def exportNet(filename, wMat, aVec):
    indMat = jnp.c_[wMat, aVec]
    jnp.savetxt(filename, indMat, delimiter=',', fmt='%1.2e')

def importNet(fileName):
    ind = jnp.loadtxt(fileName, delimiter=',')
    wMat = ind[:, :-1]  # Weight Matrix
    aVec = ind[:, -1]  # Activation functions

    # Create weight key
    wVec = wMat.flatten()
    wVec = wVec.at[jnp.isnan(wVec)].set(0)
    wKey = jnp.where(wVec != 0)[0]

    return wVec, aVec, wKey
