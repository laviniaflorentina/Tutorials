# How to clone a GitHub repository to a server and run it from your IDE

## Introduction

Let's say you want to run a project from GitHub. They usually have multiple folders, subfolders and other interconnected function files which cannot be easily managed in an Online Notebook due to memory limilations. 

The good news is if you have access to a server with better resources than your computer, you can clone the repository on that server and work on it from your IDE.  

-------------------------------
## **Step 1.** Connect to VPN 
Everytime you try to access the HPC server you need to be connected to the UNT VPN. Not sure how? Check [this tutorial](https://itservices.cas.unt.edu/services/accounts-servers/articles/cisco-anyconnect-mobility-client-vpn).

## **Step 2.** Go to the main page of the repository you want to clone. Copy the repo link. 

## **Step 3.** Open terminal/command prompt window and login to your HPC account by typing: **ssh youraccount@vis.hpc.unt.edu**, press Enter and fill in your password.

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone1.png)

I will type **pwd** to see the path where I am right now.

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone2.png)

It looks like I am in the home folder of my user so I can go ahead and clone the Git Repo here. 

**Note.** There are two options where you are able to store files: **/home/youraccount** or **/storage/scratch2/youraccount**. If you want to move to another folder use the **cd** command (see [tutorial](https://www.geeksforgeeks.org/cd-command-in-linux-with-examples/) ).

## **Step 4.** In the folder you decide to copy the repository type: **git clone git_repo_link** as in the example below.

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone3.png)

## **Step 5.** Open PyCharm and choose "**+ Create new project**".

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone4.png)

## **Step 6.** Choose the path where you want to create the project on your computer. It will automatically get named “untitled”, but you can rename it as you wish (I used _music_generation_):

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone5.png)

## **Step 7.** Notice the folder created for the project is on the left side. Look for **Tools** :arrow_right: **Deployment** :arrow_right: **Configuration**.

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone6.png)

## **Step 8.** A new window pops-up with the following tabs:

### 8.1. **Connection** - the one for connection details. 
You can select **Test Connection** (it will try to connect with the details provided). Make sure the **Root path** is either: **/storage/scratch2/youraccount** or **/home/youraccount**, depending on what you chose at **Step 4**.

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone7.png)

### 8.2. **Mappings** - the one for path matching between your computer and HPC! 
**Local path** points to the folder’s project on your computer (from **Step 6**), while **Deployment path** points to the folder’s project on HPC.

**:bangbang:Note.** It’s important to know that deployment path is referring to what’s next after the Root path from **Step 8.1**! They combined have to point to the repository on HPC! 

For example, my full path on HPC was “home/lp0348/Music-Generation-Using-Deep-Learning” but because I put “home/lp0348/” in Root path, for Deployment path only remains to write Music-Generation-Using-Deep-Learning:

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone8.png)

If you don’t know the exact path on HPC, you can go back to terminal and follow the steps:

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone9.png)

## **Step 9.** Now you should be able to see on the right side your directories on HPC and the mapped project ‘s name highlighted:

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone10.png)

9.1. If you don’t see it, go to **Tools** :arrow_right: **Deployment** :arrow_right: **Browse Remote Host** and look for the HPC connection.

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone11.png)

## **Step 10.** As you can see on the left side, the local folder’s project is still empty. The next step is to download the project from HPC to your local computer. There are two methods:

### 10.1. Synchronize with server: 
**Tools** :arrow_right: **Deployment** :arrow_right: **Sync with Deployed to …**

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone12.png)

It might ask to choose the server to sync with:

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone13.png)

The next window pop-up will look like the below. If you directly click on the first forward green arrow it will sync only the file that is selected in blue:

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone14.png)

In order to copy the entire project Ctrl+A/Command+A, then click on the first forward green arrow:

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone15.png)

### 10.2. Download from the server:

Make sure you select the local folder. Then, **Tools** :arrow_right: **Deployment** :arrow_right: **Download from**. It might ask to choose the server to sync with.

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone16.png)

On the bottom of the page there’s a progress bar and also the name of the file that is currently downloading: 

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone17.png)

You get notified once all the files are transferred and the local folder’s project is now populated:

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone18.png)

## **Step 11.** Select **Tools** :arrow_right: **Deployment** :arrow_right: **Automatic Upload** to make sure every modification you make it will automatically be made on the server too. It might ask to choose the server:

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/clone19.png)

At this point you have all the files from the GitHub repository both on your local computer and on HPC server. :white_check_mark:

Why to have them in both places? Because you will prefer to work on a file from PyCharm and the changes will be automatically synced with HPC. It is much more complicated to modify or work on files directly on HPC. 

Therefore, having the project in both places you work locally, but the program will run on the server’s resources.









