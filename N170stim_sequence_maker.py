import pandas as pd
import random
import os

#enter number of participant stimulus sequences needed 
n_participants = 2

n_obj = 100
n_female = 50
n_male = 50
n_rep_obj = 10
n_rep_face = 5

#relative image paths-- make sure that these images are saved in the same way on your computer
female_folder = "images/faces/female"
male_folder = "images/faces/male"
object_folder = "images/objects"

#read stimuli lists 
female_stimuli = pd.read_csv('image_stim_names/female_face_stim_names.csv')
male_stimuli = pd.read_csv('image_stim_names/male_face_stim_names.csv')
object_stimuli = pd.read_csv('image_stim_names/object_stim_names.csv')

#add relative filepaths
female_stimuli['filename'] = female_folder + '/' + female_stimuli['filename']
male_stimuli['filename'] = male_folder + '/' + male_stimuli['filename']
object_stimuli['filename'] = object_folder + '/' + object_stimuli['filename']

#make output directory
output_dir = 'N170stimulus_lists'
os.makedirs(output_dir, exist_ok=True)

#loop over participants 
for subj in range(1, n_participants + 1):
    random.seed(subj) 

    #sample original stimuli
    female_sample = female_stimuli.sample(n=n_female, random_state=random.randint(1, 100000))['filename'].tolist()
    male_sample = male_stimuli.sample(n=n_male, random_state=random.randint(1, 100000))['filename'].tolist()
    object_sample = object_stimuli.sample(n=n_obj, random_state=random.randint(1, 100000))['filename'].tolist()

    #sample repeat stimuli
    repeat_female = random.sample(female_sample, n_rep_face)
    repeat_male = random.sample(male_sample, n_rep_face)
    repeat_objects = random.sample(object_sample, n_rep_obj)

    #combine and shuffle all stimuli
    all_stimuli = female_sample + male_sample + object_sample + repeat_female + repeat_male + repeat_objects
    random.shuffle(all_stimuli)

    #create dataframe
    df = pd.DataFrame({'name': all_stimuli})

    #add repeat status to repeats (so repeats do not appear before originals)
    seen = set()
    repeat_status = []
    for fname in df['name']:
        if fname in seen:
            repeat_status.append('repeat')
        else:
            repeat_status.append('original')
            seen.add(fname)
    df['repeat_status'] = repeat_status

    #add category 
    def get_type(path):
        return 'object' if 'object' in path else 'face'

    df['stim_type'] = df['name'].apply(get_type)

    #save sequence csvs 
    outpath = os.path.join(output_dir, f'sub-{subj:03}_N170_stimlist.csv')
    df.to_csv(outpath, index=False)

print(f"Stimulus sets saved in: {output_dir}/")
