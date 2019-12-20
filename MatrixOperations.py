def VecByConst(matrix,k):
    ret =[]
    for i in range(len(matrix)):
        tmp = matrix[i] * k
        ret.append(tmp)

    return ret





def AddVecToMatrix(matrix,vec):
    ret = []
    for i in range(len(matrix)):
        tmp = []
        for j in range(len(matrix[0])):
            tmp.append(matrix[i][j] + vec[j])
        ret.append(tmp)

    return ret



def Transpose(matrix):
    ret = []
    for i in range(len(matrix[0])):
        tmp = []
        for j in range(len(matrix)):
            tmp.append(0)
        ret.append(tmp)

    for i in range(len(matrix[0])):
        for j in range(len(matrix)):
            ret[i][j] = matrix[j][i]

    return ret

def VecByVecMultiplication(vec1,vec2):
    if len(vec1) != len(vec2):
        return vec1
    ret = []
    for i in range(len(vec1)):
        ret.append(vec1[i] * vec2[i])

    return ret

def SubVecToVec(vec1,vec2):
    if len(vec1) != len(vec2):
        return vec1
    ret = []
    for i in range(len(vec1)):
        ret.append(vec1[i] - vec2[i])

    return ret

def AddVecToVec(vec1,vec2):
    if len(vec1) != len(vec2):
        return vec1
    ret = []
    for i in range(len(vec1)):
        ret.append(vec1[i] + vec2[i])

    return ret

def MatrixByVector(matrix, vector):
    # Function takes a matrix represented by list of lists
    # And vector represented by list and multiply them with LA principles
    result = []
    for i in range(len(matrix)):
        value = 0
        for j in range(len(matrix[i])):
            value += vector[j] * matrix[i][j]
        result.append(value)

    return result
