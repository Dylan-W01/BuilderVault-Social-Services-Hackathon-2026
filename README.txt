Step 0:
Do this on bash from the root folder of the project:
pip install Faker==25.0.0
pip install pandas==2.2.2
pip install numpy==1.26.4
pip install pyarrow==pip install
pip install matplotlib==3.9.0
pip install jupyterlab==4.1.8
pip install ipykernel==6.29.4
pip install python-dateutil
pip install textdistance
(be careful about spelling with dateutil when installing)
Step 1: open a command prompt and go to the root project folder. Type:

python tracks/referral-care-coordination/generator/generate.py
cd tracks/referral-care-coordination/
python engine.py

Step 2: start local server
python -m http.server 8000
3. Go to: http://localhost:8000/mockUI-SocialServiceHackathon.html
a. Copy and paste this link in the browser. If that doesn't work, ensure you followed all steps (including installations, generating data, moving to correct directory, running engine). Also try typing http://localhost:8000 into the browser and click on the .html file from there. If it still doesn't work, try to double-click on .\tracks\referral-care-coordination\mockUI-SocialServiceHackathon.html
b. If the json file is having trouble loading, restart your computer. and repeat step 2.
