def calculate_speed(euclidean_distance, time_interval):
    meters_in_pixel = 3.14
    speed = euclidean_distance * meters_in_pixel / time_interval
    return int(speed)
