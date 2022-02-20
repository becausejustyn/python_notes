

#! in front is the same as typing it in the terminal

#check if you have it installed
!which virtualenv 
#install it
!pip3 install virtualenv

#create a venv named python_notes
!virtualenv python_notes

#if you want to use a specific version of python, e.g. 2.7
!virtualenv -p /usr/bin/python2.7 python_notes

#Activate the virtual environment
!source python_notes/bin/activate

!which python