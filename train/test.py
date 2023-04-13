import serial
print('serial ' + serial.__version__)

# Set a PORT Number & baud rate
PORT = 'COM4'
BaudRate = 9600

ARD= serial.Serial(PORT,BaudRate)


def Decode(A):
    A = A.decode()
    A = str(A)
    if A[0]=='Q':                       #첫문자 검사
        if (len(A)==11):                #문자열 갯수 검사
            Ard1=int(A[1:5])
            Ard2=int(A[5:9])
            result= [Ard1,Ard2]
            return result
        else : 
            print ("Error_lack of number _ %d" %len(A))
            return False
    else :
        print ("Error_Wrong Signal")
        return False
    
def Ardread(): # return list [Ard1,Ard2]
        if ARD.readable():
            LINE = ARD.readline()
            code=Decode(LINE) 
            print(code)
            return code
        else : 
            print("읽기 실패 from _Ardread_")


while (True):
    Ardread()