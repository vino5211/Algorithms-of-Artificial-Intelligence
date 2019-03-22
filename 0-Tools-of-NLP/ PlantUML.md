# PlantUML

## Sequence Diagram
+ 你可以用 -> 来绘制参与者之间传递的消息，而不必显式地声明参与者。
你也可以使用”-->” 绘制一个虚线箭头。
另外，你还能用”<-” 和”<--”，这不影响绘图，但可以提高可读性。注意：仅适用于时序图，对于其
它示意图，规则是不同的。
@startuml
Alice -> Bob: Authentication Request
Bob --> Alice: Authentication Response
Alice -> Bob: Another authentication Request
Alice <-- Bob: another authentication Response
@enduml

## 用例图

## 类图

