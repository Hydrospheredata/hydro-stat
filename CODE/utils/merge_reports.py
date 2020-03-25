import shutil

# from lib import Path

outpout_file = '../outputs/Continuous/merge.log'
all_files = [('../outputs/Continuous/abalone_rings.log'),
             ('../outputs/Continuous/abalone_sex.log'),
             ('../outputs/Continuous/absenteeism_day.log'),
             ('../outputs/Continuous/absenteeism_month.log'),
             ('../outputs/Continuous/absenteeism_pet.log'),
             ('../outputs/Continuous/absenteeism_reason.log'),
             ('../outputs/Continuous/absenteeism_season.log'),
             ('../outputs/Continuous/atom.log'),
             ('../outputs/Continuous/auto_cylinders.log'),
             ('../outputs/Continuous/auto_origin.log'),
             ('../outputs/Continuous/backnote.log'),
             ('../outputs/Continuous/bike_holiday.log'),
             ('../outputs/Continuous/bike_season.log'),
             ('../outputs/Continuous/bike_weather.log'),
             ('../outputs/Continuous/chainlink.log'),
             ('../outputs/Continuous/engy_time.log'),
             ('../outputs/Continuous/bike_holiday.log'),
             ('../outputs/Continuous/bike_weather.log'),
             ('../outputs/Continuous/bike_weekend.log'),
             ('../outputs/Continuous/lsun.log'),
             ('../outputs/Continuous/Hepta.log'),
             ('../outputs/Continuous/golf_ball.log'),
             ('../outputs/Continuous/engy_time.log'),
             ('../outputs/Continuous/bike_season.log'),
             ('../outputs/Continuous/chainlink.log'),
             ('../outputs/Continuous/atom.log'),
             ]

with open(outpout_file, "wb") as wfd:
    for f in all_files:
        with open(f, "rb") as fd:
            shutil.copyfileobj(fd, wfd, 1024 * 1024 * 10)
