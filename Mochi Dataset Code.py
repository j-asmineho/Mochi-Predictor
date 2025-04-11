import pandas as pd
import random
from datetime import time
import json

# Load JSON rules
with open('Mochi.json') as f:
    activities_rules = json.load(f)

# Helper functions
def time_to_float(t):
    return t.hour + t.minute/60

def generate_row():
    # Weighted activity selection
    activity_rule = random.choices(
        activities_rules,
        weights=[r['probability'] for r in activities_rules]
    )[0].copy()  # Create a copy to avoid modifying original
    
    # Handle day constraint
    if 'day' in activity_rule:
        if isinstance(activity_rule['day'], int):
            day_num = activity_rule['day']
        else:
            day_num = random.choice(activity_rule['day'])
    else:
        day_num = random.randint(1, 7)
    day = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][day_num-1]
    
    # Generate time within specified range (with realistic distributions)
    time_range = activity_rule['time_range']
    if activity_rule['activity'] == 'Sleeping':
        # Bimodal distribution for sleeping (night + naps)
        if random.random() < 0.7:  # 70% night sleep
            time_float = random.uniform(22, 6) % 24  # 10pm-6am
        else:  # 30% daytime naps
            time_float = random.uniform(13, 16)  # 1pm-4pm
    else:
        time_float = random.uniform(time_range[0], time_range[1])
        if time_float >= 24: time_float -= 24
    
    # People home logic
    people_min = activity_rule.get('people_home_min', 0)
    people_max = activity_rule.get('people_home_max', 7)
    
    # Adjust based on time/day
    if time_float < 8 or time_float > 20:  # Night hours
        people_home = random.randint(max(1, people_min), min(3, people_max))
    elif day_num in [6,7]:  # Weekend
        people_home = random.randint(max(2, people_min), people_max)
    else:  # Weekday daytime
        people_home = random.randint(people_min, min(2, people_max))
    
    # Location handling (some locations more likely than others)
    location_weights = {
        'Living room': 40,
        'Kitchen': 30,
        'Jasmine\'s room': 15,
        'Parent\'s room': 10,
        'Bathroom': 5,
        'Front yard': 20,
        'Mailbox': 5,
        'Lily\'s room': 5
    }
    valid_locations = activity_rule['location']
    weights = [location_weights[loc] for loc in valid_locations]
    location = random.choices(valid_locations, weights=weights)[0]
    
    # Weather with seasonal patterns
    weather_probs = {
        'Sunny': 0.6,
        'Cloudy': 0.25,
        'Rainy': 0.1,
        'Snowy': 0.05
    }
    if 'weather' in activity_rule:
        valid_weather = activity_rule['weather']
        weights = [weather_probs[w] for w in valid_weather]
        weather = random.choices(valid_weather, weights=weights)[0]
    else:
        weather = random.choices(list(weather_probs.keys()), weights=weather_probs.values())[0]
    
    # Mood based on activity and other factors
    mood_rules = {
        'Sleeping': ['Sleepy', 'Lazy'],
        'Begging for food': ['Excited', 'Savage'],
        'Zoomies': ['Excited'] if people_home > 3 else ['Excited', 'Savage'],
        'Playing fetch': ['Excited'],
        'Barking': ['Annoyed'] if random.random() < 0.7 else ['Scared'],
        'Cutting nails': ['Annoyed', 'Scared'],
        'Shower': ['Annoyed'],
        'Eating': ['Excited'],
        'On the couch': ['Lazy'] if time_float > 20 else ['Excited'],
        'Pooping': ['Relieved'],
        'Peeing': ['Relieved'],
        'Haircut': ['Savage', 'Annoyed'],
        'Walking': ['Energetic', 'Excited'],
        'Trick': ['Excited']
    }
    mood = random.choice(mood_rules[activity_rule['activity']])
    
    # Trigger logic
    trigger_map = {
        'Begging for food': ['Someone cooking', 'Someone eating'],
        'Sleeping' : ['Mochi tired'],
        'Zoomies': ['Just pooped', 'Shower', 'Playing fetch'],
        'Playing fetch': ['Someone throws toy'],
        'Barking': ['Hears doorbell', 'Someone comes home', 'Someone goes downstairs'],
        'Cutting nails': ['Routine grooming'],
        'Shower': ['Bath day'],
        'Haircut': ['Hair too long'],
        'Eating': ['Meal time'],
        'On the couch': ['Mom sits down'],
        'Pooping': ['Morning walk', 'Ate food'],
        'Peeing': ['Bored'],
        'Walking': ['Bored'],
        'Trick': ['Treat']
    }
    trigger = random.choice(trigger_map[activity_rule['activity']]) if activity_rule['activity'] in trigger_map else None
    
    # Reward given - special cases first
    if activity_rule['activity'] in ['Cutting nails', 'Shower', 'Haircut', "Trick"]:
        reward = 1
    elif 'reward_given' in activity_rule:
        reward = activity_rule['reward_given']
    else:
        reward = 0
    
    # Duration logic
    duration_rules = {
        'Sleeping': lambda: random.randint(30, 480),
        'On the couch': lambda: random.randint(15, 180),
        'Walking': lambda: random.randint(20, 60),
        'Playing fetch': lambda: random.randint(10, 30),
        'Zoomies': lambda: random.randint(5, 15),
        'Eating': lambda: random.randint(5, 20),
        'Cutting nails': lambda: random.randint(10, 30),
        'Shower': lambda: random.randint(15, 45),
        'Haircut': lambda: random.randint(20, 60),
        'Barking': lambda: random.randint(1, 5),
        'Trick': lambda: random.randint(1, 5)
    }
    duration = duration_rules.get(activity_rule['activity'], lambda: random.randint(1, 10))()
    
    return {
        'Day': day,
        'Time': round(time_float, 1),
        'Duration_minutes': duration,
        'Location': location,
        'Weather': weather,
        'People_home': people_home,
        'Mood': mood,
        'Trigger': trigger,
        'Reward_given': reward,
        'Activity': activity_rule['activity']
    }

# Generate dataset
data = [generate_row() for _ in range(500)]
df = pd.DataFrame(data)

# Sort by day and time
df['Day_num'] = df['Day'].apply(lambda x: ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'].index(x))
df = df.sort_values(['Day_num', 'Time'])
df = df.drop('Day_num', axis=1)

# Save to CSV
df.to_csv('mochi_activities.csv', index=False)
print("Dataset generated with 500 rows!")


