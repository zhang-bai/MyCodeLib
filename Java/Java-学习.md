---
title: Java 学习
---



[toc]



# 一、基本概念

#### 1.执行Java 脚本

```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello World");
    }
}
```

```
$ javac HelloWorld.java
$ java HelloWorld
Hello World
```

**执行命令解析：**

以上我们使用了两个命令 **javac** 和 **java**。

**javac** 后面跟着的是java文件的文件名，例如 HelloWorld.java。 该命令用于将 java 源文件编译为 class 字节码文件，如： **javac HelloWorld.java**。

运行javac命令后，如果成功编译没有错误的话，会出现一个 HelloWorld.class 的文件。

**java** 后面跟着的是java文件中的类名,例如 HelloWorld 就是类名，如: java HelloWorld。

**注意**：java命令后面不要加.class。



#### 2.基本语法

以上述代码为例:

![img](https://www.runoob.com/wp-content/uploads/2013/12/662E827A-FA32-4464-B0BD-40087F429E98.jpg)