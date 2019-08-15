import easygui

def mat_factorization(R, P, Q, K, steps=10000, alpha=0.002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        #print step
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] >= 0 and i>=j:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.000001:
            break
    return P, Q.T


def read_words(filename):

    last = ""
    with open(filename) as inp:
        print(filename)
        while True:
            buf = inp.read(10240)
            if not buf:
                break
            words = (last+buf).split()
            last = words.pop()
            for word in words:
                yield word
        yield last
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
if __name__ == "__main__":

    import numpy

    i=0
    j=0
    missing=0
    numg=-1

    filename=easygui.fileopenbox()


    for word in read_words(filename):
        if(is_number(word)):
            if(i==0):
                numg=int(word)
                break
            else:
                print("Wrong format")
                exit()
    R = numpy.zeros(shape=(numg, numg))


    for word in read_words(filename):
        if(is_number(word)):
            if(i==0):
                continue
            #print((word))
            R[i-1][j]=(float)(word)
            j=j+1
        else:
            #print(word.__len__())
            if(word.__len__()>1):
                i=i+1
                j=0
            else:

                R[i-1][j]=-1
                missing=missing+1
                j=j+1
    print (missing)

    N = len(R)
    M = len(R[0])
    K = numg

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    nP, nQ = mat_factorization(R, P, Q, K)

    nR = numpy.dot(nP, nQ.T)

    print (nR)
    Result = numpy.zeros(shape=(numg, numg))


    for i in range(len(R)):
        for j in range(len(R[0])):
            if(R[i][j]==-1):
                Result[i][j]=nR[i][j]
            else:
                Result[i][j]=R[i][j]

    printed=""
    i=0
    j=0



    f=open(filename[:-4]+"CompletedMatfact.dis","w+")
    printed=""


    i=0
    j=0
    for word in read_words(filename):
        if(is_number(word)):
            if(i==0):
                printed = printed + word
                continue
            #print((word))
            R[i-1][j]=(float)(word)
            printed = printed + " " + word
            j=j+1
        else:
            #print(word.__len__())
            if(word.__len__()>1):
                printed = printed + "\n" + word
                i=i+1
                j=0
            else:

                R[i-1][j]=-1
                printed = printed + " " + str(round(Result[i-1][j],5))
                missing=missing+1
                j=j+1


    f.write(printed)
    f.close

'''    
    for word in read_words('realDisEdit50.dis'):


        if(is_number(word)==False and word!="."):
            printed=printed+"\n"+word
            i=i+1
        elif(word!="."):
            if(float(word)>10):
                printed=printed+"\n"+word
            else:
                if(i==len(R)):
                    break
                while(j<i):
                    #print(i,j,R[i][j])
                    x=R[i][j]
                    if(x==-1):
                        x=round(nR[i][j],5)
                    printed=printed+" "+str(x)
                    j=j+1
                j=0

    f=open("edited76forDambe.dis","w+")

    f.write(printed)
    f.close





'''








    #print nR
