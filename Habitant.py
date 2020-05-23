class habitant():

    def __init__(self, boundx, boundy):
        import random
        # Zufallsstartbelegung mit Abstand 20 von Rand
        self.x_value = random.randint(20,boundx-20) 
        self.y_value = random.randint(20,boundy-20)
        # 0=susceptible 1=infected 2=recovered
        self.condition = 0
        self.infectticker = 0

    
    def move(self, boundx, boundy):
            from random import choice
        # Führt Schritte aus, bis der Pfad die angegebene Länge erreicht hat.
            x_direction = choice([1, -1])
            x_distance = choice([0, 1, 2, 3, 4])
            x_step = x_direction * x_distance
            y_direction = choice([1, -1])
            y_distance = choice([0, 1, 2, 3, 4])
            y_step = y_direction * y_distance
            
            # Berechnet den nächsten x- und y-Wert.
            self.x_value += x_step 
            self.y_value += y_step
            
            if self.x_value < 1 or self.x_value > boundx-1:
                self.x_value -= x_step
                
            if self.y_value < 1 or self.y_value > boundy-1:
                self.y_value -= y_step