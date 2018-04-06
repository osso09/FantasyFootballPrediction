# FantasyFootballPrediction

We are investigating methods for improving fantasy football prediction models.

This project is expanding on work provided by https://github.com/romanlutz/fantasy-football-prediction.

It uses the An API to retrieve and read NFL Game Center JSON data. 
This API can be found at https://github.com/BurntSushi/nflgame


Basic setup: 
This project uses python 2.7. Download and instructions for python can be found at https://www.python.org/ .

Installing dependencies:<br/>
$ sudo apt-get install pip<br/>
$ pip install --upgrade python<br/>
$ sudo pip install numpy<br/>
$ sudo pip install pybrain<br/>
$ sudo pip install scipi<br/>
$ sudo pip install sklearn<br/>
$ sudo pip install matplotlib<br/>
$ pip install nflgame<br/>

The following updates data for the datasets used for the machine learning from nflgame:<br/>
$ sudo nflgame-update-players<br/>
$ python get_data.py<br/>
$ python create_datasets.py<br/>
$ python neural_net.py<br/>
  
  
