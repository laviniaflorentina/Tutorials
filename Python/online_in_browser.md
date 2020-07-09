# How to get started with Google Colaboratory & Jupyter Notebook

Content
- [Introduction](https://github.com/laviniaflorentina/Tutorials/blob/master/Python/online_in_browser.md#introduction)
- [Google Colaboratory](https://github.com/laviniaflorentina/Tutorials/blob/master/Python/online_in_browser.md#google-colaboratory)
    - [New Notebook](https://github.com/laviniaflorentina/Tutorials/blob/master/Python/online_in_browser.md#new-notebook)
    - [Main Features](https://github.com/laviniaflorentina/Tutorials/blob/master/Python/online_in_browser.md#main-features)
    - [Runtime Environment Options](https://github.com/laviniaflorentina/Tutorials/blob/master/Python/online_in_browser.md#runtime-environment-options)
    - [Run the Notebook](https://github.com/laviniaflorentina/Tutorials/blob/master/Python/online_in_browser.md#run-the-notebook)
    - [Download the Notebook](https://github.com/laviniaflorentina/Tutorials/blob/master/Python/online_in_browser.md#download-the-notebook)
    - [Share the Notebook](https://github.com/laviniaflorentina/Tutorials/blob/master/Python/online_in_browser.md#share-the-notebook)
- [Jupyter Notebook](https://github.com/laviniaflorentina/Tutorials/blob/master/Python/online_in_browser.md#jupyter-notebook)
    - [New Notebook](https://github.com/laviniaflorentina/Tutorials/blob/master/Python/online_in_browser.md#new-notebook-1)
    - [Main Features](https://github.com/laviniaflorentina/Tutorials/blob/master/Python/online_in_browser.md#main-features-1)
-----------------------------------

# Introduction

Installing PyCharm (or any software) can be a little difficult for a newbie. Fortunately, there are many online resources to get familiar with the syntax and the features of Python before proceeding to install an IDE in the local machine.

[Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb) (“Colab”, for short) and [Jupyter Notebook](https://jupyter.org/try) are some of the most popular online platforms that allow you to write and execute Python in your browser.

# Google Colaboratory 

To get started with Colab there are two methods: 

1. **Direct link**: by accessing https://colab.research.google.com. This will open a page as shown below, and you are all set to start.

If you are not logged in with a gmail account, it will probably look like this:

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/colab1.png)

If you are already logged in with a gmail account, it will probably look like this:

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/colab2.png)

There are some tab options:

**EXAMPLES**: Contains several files designed to present different types of notebooks.

**RECENT**: Contains the last notebooks you have worked with.

**GOOGLE DRIVE**: Contains the notebooks you have in your Google Drive account.

**GITHUB**: Use this tab if you need to add notebooks from your GitHub, but you first need to connect Colab with GitHub!

**UPLOAD**: Upload notebooks from your local computer.

You can create a new notebook by clicking **NEW NOTEBOOK** at the bottom right corner. Or **CANCEL** if you do not wish to continue.

2. **Google Drive**: if you have a Google Drive account, access your drive https://drive.google.com as usual and look for the " + New " button which should be in the upper left corner as shown below:

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/colab3.png)

Click on **+ New** :arrow_right: **More** :arrow_right: **Google Colaboratory** and you are all set to start!

:pushpin: Make sure you SignIn for a better experience. 

:pushpin: Works the best in Google Chrome! 

## New Notebook

**New Notebook** option will create an _Untitled0.ipynb_ file and it will automatically be saved to your Google Drive in a folder named Colab Notebooks. The new file will look similarly with the following screenshot:

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/colab4.png)

## Main Features

I will describe each feature marked in the above screenshot. 

:one: The name of the file. The extension ".ipynb" is specifically for Python.

:two: **Code** and **Text** in both places it will create a new cell either for writing code or for adding [markdown plain text](https://www.markdownguide.org/cheat-sheet/) cell.

:three: Represents an Input code cell. Notice it begins with ":arrow_forward:". You can click on that icon to run the code in the cell or you can use the shortcut CTRL+Enter/Command+Enter.

:four: Represents an Output of the code cell. Notice it begins with a special icon. You can click on that icon to clear the output of the code cell.

:five: Other Cell - related features are:
![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/colab6.png)

You can: _Move Up or Move Down the Cell, Link to Cell, Add comments for a Cell, Delete the Cell_, but you can also Copy Cell (CTRL+C/Command+C), Cut Cell (CTRL+X/Command+X), Paste Cell (CTRL+V/Command+V) & Edit Cell (double click on it).

If you click on the settings button you can see the following control panel:

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/colab7.png)

**Site.** Has options for the whole website with regard to the Theme (_Light, Dark, Adaptive_).

**Editor.** Has options for the editor with regard to the Text Coloring (_Default, Monokai, Intense and Synthwave84_), Key Bindings (_Default, VIM & Classic_), Font size, etc.

**Colab Pro.** They present the benefits of getting a subscrption for a Pro account. With Colab Pro you have priority access to our fastest GPUs. 

**Miscellaneous.** Power level is an April fools joke feature that adds sparks and combos to cell editing. See [their post on Twitter](https://twitter.com/GoogleColab/status/1112708634905964545).

## Runtime Environment Options

Neural networks, for example, are a common task that needs more powerful resources then an usual program. Colab is offering free access to GPU and TPU resources. You can change the “Hardware accelerator” to GPU or TPU based on your code complexity. Read more about [Hardware parts in training a Neural Network](). 

Click on **Runtime** dropdown menu. 

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/colab8.png)

Select **Change runtime type** and another window will pop-up:

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/colab9.png)

You can choose either one, depending on your needs, then run the entire program.

## Run the Notebook

There are different options for running the code in browser which is a huge advantage in contrast with an IDE.

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/colab8.png)

You can select:

1. **Run all** and it will run all the cell in order from the first one to the last one.

2. **Run before** and it will run the cell before the one selected.

3. **Run the focused cell** and it will run the selected cell.

4. **Run selection** and it will run only the selected part from a cell.

5. **Run after** and it will run all the cells one by one.

From the same dropdown list you can:

1. **Interrupt the execution** of the program.

2. **Restart runtime** command will drop your current backend memory, like data you have downloaded, packages & libraries you have installed, etc.

3. **Restart and run all** command behaves like the previous command and, additionally, it runs the entire program after restart.

4. **Factory reset runtime** command is similar with the Restart, but more powerful.

## Download the Notebook

If you want to download the notebook file: click on **File** (top left corner) :arrow_right: _download .ipynb_ or _download .py_. See screenshot below:

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/colab10.png)

## Share the Notebook

If you want to share the notebook file: click on **Share** (top right corner), and choose one of the sharing options:

1. By e-mail invite: type in the person e-mail address:

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/colab11.png)

2. By link sharing. Here, you can customize the viewer rights: _viewer/ commenter/ editor_, as well as who might have access: _Restricted_ (only people with the link) or _Anyone on the internet with the link_.

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/colab12.png)

# Jupyter Notebook

To get started with Jupyter Notebook you can click the link https://cocalc.com/doc/jupyter-notebook.html. 

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/jup1.png)

There are two options: **Run Jupyter Now** or **Sign In** with a social network (_Facebook, GitHub, etc._). For a better experience it's recommended to be Signed In.

After Sign In, click on the button **Projects** (top left corner) to create a Project file - in order to manage all .py and dataset files within that project. 

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/jup2.png)

Give a name to the Project (I used _Getting_started_), then click on :arrow_right: **+ Create New Project...** :arrow_right: again **+ Create New Project**. This is the main folder for the project.

## New notebook

Next, in order to get a new notebook it asks you to create or upload a file:

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/jup3.png)

Click on _+ Create or Upload Files_. 

I will start with creating a new file by giving it a name (I used _first_jupyter_file_) and selecting its type as a **Jupyter Notebook**.

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/jup4.png)

Next, you need to choose the kernel (the programming language) that you want to work with. I will go with _Python 3_. 

If you typically use the same programming language all the time, then you can select "Do not ask, instead default to your most recent selection (..)" so that the next time it will open a file of the same type you created the last time. 

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/jup5.png)

Once you select the programming language it will finally open a new notebook!

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/jup6.png)

## Main Features

I will describe each feature marked in the above screenshot. 

:one: The name of the file. The extension ".ipynb" is specifically for Python.

:two: The name of the project where your notebook file is.

:three: Represents an Input code cell. To run the code in the cell you can use the shortcut CTRL+Enter/Command+Enter.

:four: Represents an Output of the code cell. 

:five: Other Cell - related features:

:mag_right: Zoom Out/Zoom In -- for font size; :scissors: Cut, :clipboard: Copy, :page_facing_up: Paste; 

:heavy_plus_sign: create a new cell; :arrow_up:/:arrow_down: move the cell up/down; :arrow_forward: go run the next cell ; :black_medium_small_square: interrupt execution; :arrows_counterclockwise: restart the entire notebook; :fast_forward: run next cells.

:six: Notebook presentation options.
