---
layout: post
title:  Echo Server cpp
date:   2023-09-09 
description: Intro to socket programming in C++
tags: networking, C++
categories: 
---

# Intro Socket Programming in C

Hello! It has been a while since my last post. I have been busy with job applications and switching jobs the last few weeks and haven't set aside enough time for this. From now on I'll make sure to post every week I'll say Monday even if what I am working on is not in any kind of finished state I'll just put my progress or talk about something completely different. Like a filler episode in anime.

This week I am getting started with C++ and neworking/socket programming. I want to learn C++ because I said on my resume and some job applications that I know it and I want to also improve my networking/socket programming skills. The end goal is to create a something more interesting like a packet sniffer (like libpcap) or a blockchain network but since I am just getting started this week I am getting used to socket programming in C++ and implementing an echo server and client

## Echo Server

One guide that was very helpful with getting familiar with socket programming in C was [Beej's guide](https://beej.us/guide/bgnet/html/split/slightly-advanced-techniques.html#blocking). I knew the overall structure for a echo server. I would need:
1. Accept connection from client socket
2. Have multi-threading capability to allow the server to handle multiple clients simultaneously
3. Ability to read message from client and send message back
4. Client side should have some kind of command line program to allow client to type message and receive echo

Some parts of this are in the server side and some parts are on the client side. I will start by going throught the server side.

## Server
First some variables I initialized at the beginning of the program that I will use throughout server.cpp:


```c
int status;
int sockfd, new_fd;
int port_num;
struct sigaction sa;
socklen_t addr_size;
struct sockaddr_storage their_addr;
struct addrinfo hints;
struct addrinfo *res, *p;
socklen_t sin_size;
char s[INET6_ADDRSTRLEN];
char ipstr[INET6_ADDRSTRLEN];
memset(&hints, 0, sizeof hints);
hints.ai_family = AF_UNSPEC; // AF_INET or AF_INET6 to force version
hints.ai_socktype = SOCK_STREAM;
int yes = 1;
```

The first step is to get the address info of the server and that is done with the function 'getaddrinfo':


```c
if ((status = getaddrinfo(NULL, "3490", &hints, &res)) != 0){
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(status));
        return 1;
    };
```

This populates `res` with a linked list of potential addresses that a client can bind to in the server. The next step is to iterate through the possible addresses and initialize a socket file descriptor using one of the valid addresses.

```c
for(p = res; p!= NULL; p = p->ai_next){
        if((sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol))== -1){
            perror("server: socket");
            continue;
        }
        if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &yes,
                sizeof(int)) == -1) {
            perror("setsockopt");
            exit(1);
        }
        if (bind(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
            close(sockfd);
            perror("server: bind");
            continue;
        }

        break;
    }
```

The socket fild descriptor(`sockfd`) will be set to the first address that is valid and if there are no valid addresses the program will exit:

```c
freeaddrinfo(res);
    if(p==NULL){
        fprintf(stderr, "Failed to bind\n");
        exit(1);
    }
```

The next step is for the socket to listen to incoming clients trying to connec to the server which is done through the `listen()` function:


```c
if(listen(sockfd, BACKLOG)==-1){
        perror("listen");
        exit(1);
    };
    // waiting for connection now
    printf("server waiting for connections ... \n");
```

This code concludes the setup of the server; it now is able to listen to incoming clients. Now the server must accept clients and send echos to each client simultaneously as long as the server is active. Since the following functionality happens for the duration of the lifetime of the server, the remaining code is within a while loop: `while(1) `

The first step is to accept clients that connec to the server:


```c
sin_size = sizeof their_addr;
new_fd = accept(sockfd, (struct sockaddr *)&their_addr, &sin_size);
if (new_fd == -1) {
    perror("accept");
    continue;
}
```

The return value of the `accept` function is the filedescriptor of the client socket and can be used as a parameter in `send` and `recv` to send and receive data from the client socket. After the client is connected to the server, the server must spawn a child process to handle this client simultaneously with the parent process as well as handling requests from other clients. This is where we must use `fork()` to do this. I also send a message to the client explaining what this particular server does:

```c
!fork() // this is the child process
close(sockfd); // child doesn't need the listener
if (send(new_fd, "Hello! welcome to echo server. Type something and I'll type the exact same thing back. How exciting!!", 256, 0) == -1)
    perror("send");
```

Now that the client is connected and running in a child process, the server must do what it is intended to do: echo messages from the client. So whenever the server socket receives data from this client it will send a message with the same data back to the client.



```c
while (1){
    //printf("trying to recv from client");
    int numbytes = 0;
    char *buf[256];
    numbytes = recv(new_fd, buf, 256, 0);

    if (numbytes>0){
        printf("sever: received '%s'\n",buf);
    }
    
    if (numbytes>1){
        int bytes_sent;
        if ((bytes_sent = send(new_fd, buf, 256, 0)) == -1) {
            perror("send");
            printf("exiting");
            exit(1);
        }
    }
}
```

This concludes the logic needed on the server side. In summary, we get the address info of the server, initialize a socket file descriptor using the address, bind the socket to a specific port, listen to incoming connection requests form clients, accept connection from client, start a child process for client and send echo messages. Now we can move on to the client side logic which is very similar.
## Client

The client side will be similar to the server side but instead of listening for requests, now we are making a connection request and we need to have functionality allowing the user to connect to a particular server and make echo requests to the server. The plan is to have the client executable run like this: `./client ***server name***` and once connected to the server the user should be able to type a message and receive the echo.

First some variable intialized at the beginning of the script that will be used throughout client.cpp:



```c
int status;
int sockfd;
int numbytes;

char *buf[100];

struct addrinfo hints;
struct addrinfo *res, *p;
socklen_t sin_size;

char s[INET6_ADDRSTRLEN];
```

Make sure that the command line arguments are coming in correctly:


```c
if (argc != 2) {
    fprintf(stderr,"usage: client hostname\n");
    exit(1);
}
```


Now we get the address info of the server using the command line argument:



```c
// getting addrinfo and binding to specific port
if ((status = getaddrinfo(argv[1], "3490", &hints, &res)) != 0){
    fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(status));
    return 1;
};
```

The result is stored a linked list in res just as before. Just as before we iterate through the possible addresses and create a sock file descriptor using the first valid one and make a connection request:



```c
for(p = res; p!= NULL; p = p->ai_next){
    if((sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol))== -1){
        perror("client: socket");
        continue;
    }
    if (connect(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
        close(sockfd);
        perror("client: connect");
        continue;
    }

    break;
}
```

After the client is connected to the socket there will be a loop of the client receiving the message from the server then getting a prompt to make another echo request. If another echo request is made then another interation fo the loop happens. We will have a way for the client to break out of the loop and end the program if they type `exit` when they are prompted with another echo request.



```c
while (1) {

    if ((numbytes = recv(sockfd, buf, MAXDATASIZE-1, 0)) == -1) {
        perror("recv");
        exit(1);
    }

    buf[numbytes] = '\0';

    printf("client: received '%s' from server\n",buf);
    
    char buffer[256];
    printf("Enter a message: ");
    
    // Read a line of input from the user
    if (fgets(buffer, sizeof(buffer), stdin) == NULL) {
        printf("Error reading input.\n");
        continue;
    }

    // Remove the trailing newline character, if any
    if (buffer[strlen(buffer) - 1] == '\n') {
        buffer[strlen(buffer) - 1] = '\0';
    }

    // Exit the loop if the user types "exit"
    if (strcmp(buffer, "exit") == 0) {
        break;
    }
    //printf("%s\n", buffer);

    // send the message
    int bytes_sent;
    if ((bytes_sent = send(sockfd, buffer, 256, 0)) == -1) {
        perror("send");
        exit(1);
    }
}
```

This concludes the client side code.

## Final Thoughts

To conclude -- this was just a very short introduction into socket programming in C/C++ and I plan to go deeper into network programming. I feel I got a good understanding for setting up a server client connection and plan to do a more complicated network program next time like a network packet sniffer, or simple blockchain network. I'll be back here next week!
























