#include <Servo.h>

Servo myservo1;  // 분류대 서보모터
Servo myservo2;  // create servo object to control a servo

Servo myservo3;
Servo myservo4;

//분류대
int pos1 = 0;    // variable to store the servo position
int flag = 3; // 모드 설정

int input_data = 10; 
int output_data = 0;
float duration;
float distance = 50;

int trigPin2 = 4;
int echoPin2 = 3;

void setup() {

  Serial.begin(9600);
  pinMode(trigPin2, OUTPUT);
  pinMode(echoPin2, INPUT);
  pinMode(motor_1, OUTPUT);
  pinMode(motor_2, OUTPUT);
}

void loop() {
  if (flag == 3){ 
      analogWrite(motor_1, 255);
      digitalWrite(motor_2, 0);
      
      while(distance > 5 and flag == 3){
          digitalWrite(trigPin2, HIGH);
          delay(10);
          digitalWrite(trigPin2, LOW);
          // echoPin 이 HIGH를 유지한 시간을 저장 한다.
          duration = pulseIn(echoPin2, HIGH); 
          // HIGH 였을 때 시간(초음파가 보냈다가 다시 들어온 시간)을 가지고 거리를 계산 한다.
          distance = ((float)(340 * duration) / 10000) / 2; 
          
          if (distance<=54){
                  delay(3500);
                  flag = 0;
                  analogWrite(motor_1, 0);
                  digitalWrite(motor_2, 0);  
                  distance = 100;

            }
      }
  }
  // PC에 신호를 전송(쓰레기 값이 들어가는 일이 있어 delay 부여)
  if (flag == 0){
      flag = 1;
      Serial.println(1);
      delay(2000);
      
 
  }
  
  // PC 결과를 받아 서보 모터 제어
  if (flag == 1){
  while(Serial.available()>0)
  {
    motor_flag = Serial.read();
  }
  
  if (motor_flag == '1'){  // case1 모터 동작  
  
  myservo1.attach(8);
  myservo2.attach(9);
  myservo3.attach(12);
  myservo4.attach(11);
  
  for (pos1 = 90; pos1 < 180; pos1 += 1) 
  { 
    myservo1.write(pos1);
    myservo2.write(180-pos1);            
    delay(10);                      
  }
  delay(1000);
  for (pos1 = 179; pos1 >= 90; pos1 -= 1) 
  { 
    myservo1.write(pos1);
    myservo2.write(180-pos1);            
    delay(10);                      
  }

  myservo1.detach();
  myservo2.detach();
  myservo3.detach();
  myservo4.detach();
  motor_flag = '0';
  flag =3;
  }

  if (motor_flag == '2'){ // case2 모터 동작  
     
  myservo1.attach(8);
  myservo2.attach(9);
  myservo3.attach(12);
  myservo4.attach(11);
  
  for (pos1 = 90; pos1 > 0; pos1 -= 1) 
  { 
    myservo1.write(pos1);
    myservo2.write(180-pos1);            
    delay(10);                      
  }
  delay(1000);
  for (pos1 = 1; pos1 <= 90; pos1 += 1) 
  { 
    myservo1.write(pos1);
    myservo2.write(180-pos1);            
    delay(10);                      
  }

  for (pos1 = 90; pos1 < 180; pos1 += 1) 
  { 
    myservo3.write(pos1);
    myservo4.write(180-pos1);            
    delay(10);                      
  }
  delay(1000);
  for (pos1 = 179; pos1 >= 90; pos1 -= 1) 
  { 
    myservo3.write(pos1);
    myservo4.write(180-pos1);            
    delay(10);                      
  }
  
  myservo1.detach();
  myservo2.detach();
  myservo3.detach();
  myservo4.detach();
  motor_flag = '0';
  flag =3;
  }


  if (motor_flag == '3'){ // case2 모터 동작  
     
  myservo1.attach(8);
  myservo2.attach(9);
  myservo3.attach(12);
  myservo4.attach(11);
  
  for (pos1 = 90; pos1 > 0; pos1 -= 1) 
  { 
    myservo1.write(pos1);
    myservo2.write(180-pos1);            
    delay(10);                      
  }
  delay(1000);
  for (pos1 = 1; pos1 <= 90; pos1 += 1) 
  { 
    myservo1.write(pos1);
    myservo2.write(180-pos1);            
    delay(10);                      
  }

  for (pos1 = 90; pos1 > 0; pos1 -= 1) 
  { 
    myservo3.write(pos1);
    myservo4.write(180-pos1);            
    delay(10);                      
  }
  delay(1000);
  for (pos1 = 1; pos1 <= 90; pos1 += 1) 
  { 
    myservo3.write(pos1);
    myservo4.write(180-pos1);            
    delay(10);                      
  }

  
  myservo1.detach();
  myservo2.detach();
  myservo3.detach();
  myservo4.detach();
  motor_flag = '0';
  flag =3;
  }
  
  
  }

}
