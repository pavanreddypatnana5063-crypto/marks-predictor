import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Load the real data
df = pd.read_csv('score.csv')

# 2. Prepare Data for the Model
# X is what we use to predict (Features). Notice the double brackets!
X = df[['Hours']] 
# y is what we want to predict (Target).
y = df['Scores']  

# 3. Split the data
# We give the model 80% of the data to learn from, and hide 20% to test it later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully!\n")

# 5. Test the Model's Accuracy
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(f"Model Accuracy (R-squared): {accuracy:.2f}")
print("(A score of 1.0 is perfect, 0.0 is terrible)\n")

# 6. Make a Real Prediction!
# Let's see what the model predicts for a student who studies 9.25 hours
hours_studied = int(input("hours studied : "))
``
# Create a tiny DataFrame just for this one student
new_student_data = pd.DataFrame({'Hours': [hours_studied]})

# Now predict using that DataFrame
predicted_score = model.predict(new_student_data)

print("--- Prediction ---")
print(f"If a student studies for {hours_studied} hours...")
print(f"The model predicts a score of: {predicted_score[0]:.2f}")