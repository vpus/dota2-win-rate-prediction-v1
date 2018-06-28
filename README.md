# Dota2 Win Rate Prediction Model v1

**Dota2 Win Rate Prediction Model** is a model that we as VPGame's US research team build and trained in late 2016. The model is then used to build the project of [AlphaMao](http://www.vpgame.com/alphamao), which predicts the win rate by inputing the line up of both teams. **The model that supports the product is OUTDATED.**

The training **paused** when we rebuilt and migrate our Dota2 Data System to GCP since Novmember 2017.

The project is now **closed** due to the release of Dota Plus. This is now a learning project for training and testing purposes.

**DISCLAIMER**: this code is an early research code. What this means is:
- It's not the exact same code when we use in production. But it shares the logic.
- It may not work reliably (or at all) on your system. Especially the libraries we were using are very old.
- We tried to do some brief clean up before uploading to Github. But the code quality and documentation are still quite lacking. The variable names may look random and we may not have a plan on improving that
- There are quite a few hacks made specifically for our systems and infrastructure.

## Dependencies
We run the project using:
- Ubuntu 16.04 / Windows 10
- Python 3.5 / 3.6
- Keras 2.0.2
- Theano 0.9.0

With a newer version of `Keras` or `Theano` you may need to update some callings of functions like `Merge` since the way we are writing it may be deprecated.

## Inside the Repo ##
We provided our training script along with our model. We also provide a test script so that you can test the result of your trained model

- Training: `wrpt_train.py`
- Model: `wrpt_model.py`
- Testing: `wrpt_test.py`

There are also some other supporting training/testing files. 
- `new_combo_2.vp`: Win rate of two heroes when in the same team
- `new_advantage_against.vp`: Advantage Rating of one hero against another
- `2017-11-07_learning_data_1.vp`: Training Data that includes 54339 matches of 2017-11-07 matches. On average, there are 600,000 valid matches in one day. 
- `train6_v707_new_temp.weights`: Sample training result using the provided training data

## To Train ##
```
cd dir_to_project/
```
```
python wrpt_train.py
```
or if you have python2 as default python
```
python3 wrpt_train.py
```

When an epoch of training is finished, you should be able to see a screen like this:
```
54339/54339 [==============================] - 73s - loss: 0.6745 - hid_layer_loss: 0.6733 - another_output_loss: 0.0024 - hid_layer_acc: 0.5820 - another_output_acc: 0.2177 - val_loss: 0.6889 - val_hid_layer_loss: 0.6883 - val_another_output_loss: 0.0012 - val_hid_layer_acc: 0.5091 - val_another_output_acc: 0.2000
```
Where `hid_layer_acc` is the overall accuracy. In this case, it's `0.5820`. As you can see we trained this model using 54339 matches of 2017 Nov. 07th. 

## To Test ##
```
python wrpt_test.py #hero_id_1 #hero_id_2 #hero_id_3 #hero_id_4 #hero_id_5 #hero_id_6 #hero_id_7 #hero_id_8 #hero_id_9 #hero_id_10
```
where `hero_id_#` (1-5) are the heroes of Radiant and `hero_id_#` (6-10) are the heroes of Dire. 

It takes time to load the weight into the model and to build the Neural Network. After it's done you should be able to see a result like this:
```
[ 0.351208    0.64879197]
```
Where the first value is the win rate of **Dire** and the second is the win rate of **Radiant**. Keep in mind that the order may look reversed to you. This is because in Dota 2 Radiant is considered as **1**. 

In this case, we simply input 
```
python wrpt_test.py 1 2 3 4 5 6 7 8 9 10
```
which means Radiant has a line up of 
- Antimage
- Axe
- Bane
- Bloodseeker
- Crystal Maiden

and Dire has a line up of
- Drow Ranger
- Earthshaker
- Juggernaut
- Mirana
- Morphling

The win rate of Radiant (before playing the game) is 64.9%

We also offered overwrite code inside `wrpt_test.py` (Line #24, #25):
```
# rad_squad = [1,2,3,4,5]
# dire_squad = [6,7,8,9,120]
```
You can uncomment these two lines and set the `hero_id`s there as well.

[Hero ID look up](http://api.steampowered.com/IEconDOTA2_570/GetHeroes/v1?key=E0185D74028EF0F977C7F4657D9EFFC5) 

Please don't abuse the API Key, if the api key is banned we are not going to update it.

## Understand the Accuracy ##
(WIP)

## Some Improvements we made but not shown in the code ##
- The model doesn't know that the order of a team doesn't affect the result. For example, if you input `1 2 3 4 5` and `5 4 3 2 1` for Radiant, the result will be different (but really close). What we did is for each match input, `[h1 h2 h3 h4 h5]`, we created a group of "cloned" matches with same heroes, but in different order. After a large scale of training, the model will learn that the order doesn't matter. Another benifit of doing this is this will create a lot training data and help your model to learn better with limited amount of match data.

## Common Questions ##
### Where to get more training data?
We rebuilt our Dota2 Data System and now it's under [VArena](https://github.com/fundata-varena). It's still in Beta and currently it cannot support the sufficient data. It will be online soon.

On the other hand, if you don't mind old data, we are also planning to host our former training data (includes 100,000,000+ matches before 2017 July) this summer. For now you can download [these 599868 matches data](http://usa.vpgame.com/tempo_storm/700_learning_data_2.pickle) to play around. This should be enough to make your model to reach a 58+% overall accuracy and 70%+ cross entropy accuracy. 

## Discussion ##
If you have anything you'd like to discuss. Feel free to shoot an email to `gan_bing@vpgame.cn` or `shimakaze@vpesports.com`. 

If you would like to join us for more fancinating game related AI project, check [this document](https://github.com/vpus/policies/blob/master/AI-Project%20_%20Machine%20Learning%20Researcher%252FEnigneer%20Problem.pdf)

