cd datasets
# download the gazefollow preprocess file
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HOu4DtrAx8Xi8xvMH21YRM1XpJS4cRMk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1HOu4DtrAx8Xi8xvMH21YRM1XpJS4cRMk" -O gazefollow_preprocess.zip && rm -rf /tmp/cookies.txt
# download the videotargetattention preprocess file
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1LbL_B35KnCGjAT5OOwUuLg9S60F-Mhw2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LbL_B35KnCGjAT5OOwUuLg9S60F-Mhw2" -O videotargetattention_preprocess.zip && rm -rf /tmp/cookies.txt
unzip gazefollow_preprocess.zip && rm gazefollow_preprocess.zip
unzip videotargetattention_preprocess.zip && rm videotargetattention_preprocess.zip