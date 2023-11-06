import pandas as pd

Quizdf = pd.read_csv("Quiz.csv", delimiter=",", usecols=['Question', 'QuestionNo'])
Answer = pd.read_csv("Quiz.csv", delimiter=",", usecols=['AnswerOption'])
#print(Quizdf)
print("Enter your name")
Name = input()
print("HI",Name,"!!!! You can attempt this quiz only 2 times")
print("Read the Questions carefully & Select Your Option....")
print("You will get 1mark for each correct answer and 0 for Wrong")



def PythonQuiz():
    count = 0
    total = 10
    for i in range(1, 11):
        Question = (Quizdf['Question'][i])
        Solution = (Answer['AnswerOption'][i])
        print(Question)
        OptionInput = input()
        if(OptionInput == Solution):
            print("correct")
            count+=1
        else:
            print("wrong","the correct option was",Solution)

    print("Dear",Name,"THANKYOU FOR ATTEMPTING THE QUIZ")
    print("your total score is",count,"out of",total)

def main():
    #OptionInput = input()
    x = PythonQuiz()
    print(x)


if __name__=="__main__":
    main()
