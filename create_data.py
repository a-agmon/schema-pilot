import pandas as pd

# Define the data for the CSV
data = [
    ['Customers', 'main table that holds customer data', 'CustomerID', 'numeric', 'unique identifier for customer'],
    ['Customers', 'main table that holds customer data', 'Name', 'varchar', 'customer\'s full name'],
    ['Customers', 'main table that holds customer data', 'Email', 'varchar', 'customer contact email'],
    ['Customers', 'main table that holds customer data', 'PhoneNumber', 'varchar', 'primary contact number'],
    ['Customers', 'main table that holds customer data', 'RegionID', 'numeric', 'references Regions table'],
    ['Customers', 'main table that holds customer data', 'RegistrationDate', 'date', 'customer registration date'],
    ['Customers', 'main table that holds customer data', 'CustomerType', 'varchar', 'customer category (individual, business)'],
    ['Regions', 'geographical regions for customer segmentation', 'RegionID', 'numeric', 'unique identifier for region'],
    ['Regions', 'geographical regions for customer segmentation', 'RegionName', 'varchar', 'name of the region'],
    ['Regions', 'geographical regions for customer segmentation', 'Country', 'varchar', 'country of the region'],
    ['Regions', 'geographical regions for customer segmentation', 'TimeZone', 'varchar', 'standard time zone'],
    ['Interactions', 'log of all customer interactions', 'InteractionID', 'numeric', 'unique identifier for interaction'],
    ['Interactions', 'log of all customer interactions', 'CustomerID', 'numeric', 'references Customers table'],
    ['Interactions', 'log of all customer interactions', 'AgentID', 'numeric', 'references Agents table'],
    ['Interactions', 'log of all customer interactions', 'InteractionType', 'varchar', 'type of interaction (call, chat, email)'],
    ['Interactions', 'log of all customer interactions', 'StartTime', 'datetime', 'interaction start time'],
    ['Interactions', 'log of all customer interactions', 'EndTime', 'datetime', 'interaction end time'],
    ['Interactions', 'log of all customer interactions', 'Duration', 'numeric', 'interaction length in seconds'],
    ['Interactions', 'log of all customer interactions', 'Channel', 'varchar', 'communication channel used'],
    ['Agents', 'information about call center representatives', 'AgentID', 'numeric', 'unique identifier for agent'],
    ['Agents', 'information about call center representatives', 'FirstName', 'varchar', 'agent\'s first name'],
    ['Agents', 'information about call center representatives', 'LastName', 'varchar', 'agent\'s last name'],
    ['Agents', 'information about call center representatives', 'HireDate', 'date', 'date agent was hired'],
    ['Agents', 'information about call center representatives', 'DepartmentID', 'numeric', 'references Departments table'],
    ['Agents', 'information about call center representatives', 'Email', 'varchar', 'agent\'s work email'],
    ['Agents', 'information about call center representatives', 'Status', 'varchar', 'current work status (active, on leave)'],
    ['Departments', 'organizational departments in call center', 'DepartmentID', 'numeric', 'unique identifier for department'],
    ['Departments', 'organizational departments in call center', 'DepartmentName', 'varchar', 'name of the department'],
    ['Departments', 'organizational departments in call center', 'Manager', 'varchar', 'name of department manager'],
    ['Departments', 'organizational departments in call center', 'Location', 'varchar', 'physical location of department'],
    ['Tickets', 'customer support tickets', 'TicketID', 'numeric', 'unique identifier for ticket'],
    ['Tickets', 'customer support tickets', 'CustomerID', 'numeric', 'references Customers table'],
    ['Tickets', 'customer support tickets', 'AgentID', 'numeric', 'references Agents table'],
    ['Tickets', 'customer support tickets', 'CreationDate', 'datetime', 'ticket creation time'],
    ['Tickets', 'customer support tickets', 'LastUpdated', 'datetime', 'most recent ticket update'],
    ['Tickets', 'customer support tickets', 'Status', 'varchar', 'current ticket status (open, closed, in progress)'],
    ['Tickets', 'customer support tickets', 'Priority', 'varchar', 'ticket priority level'],
    ['Tickets', 'customer support tickets', 'Category', 'varchar', 'type of issue (technical, billing)'],
    ['TicketNotes', 'additional notes on support tickets', 'NoteID', 'numeric', 'unique identifier for note'],
    ['TicketNotes', 'additional notes on support tickets', 'TicketID', 'numeric', 'references Tickets table'],
    ['TicketNotes', 'additional notes on support tickets', 'AgentID', 'numeric', 'references Agents table'],
    ['TicketNotes', 'additional notes on support tickets', 'NoteText', 'text', 'content of the note'],
    ['TicketNotes', 'additional notes on support tickets', 'CreationTime', 'datetime', 'when note was added'],
    ['CustomerFeedback', 'satisfaction surveys and feedback', 'FeedbackID', 'numeric', 'unique identifier for feedback'],
    ['CustomerFeedback', 'satisfaction surveys and feedback', 'InteractionID', 'numeric', 'references Interactions table'],
    ['CustomerFeedback', 'satisfaction surveys and feedback', 'SatisfactionScore', 'numeric', 'customer satisfaction rating'],
    ['CustomerFeedback', 'satisfaction surveys and feedback', 'FeedbackText', 'text', 'detailed customer comments'],
    ['CustomerFeedback', 'satisfaction surveys and feedback', 'FeedbackDate', 'date', 'date feedback was submitted'],
    ['AgentPerformance', 'tracking agent performance metrics', 'PerformanceID', 'numeric', 'unique identifier for performance record'],
    ['AgentPerformance', 'tracking agent performance metrics', 'AgentID', 'numeric', 'references Agents table'],
    ['AgentPerformance', 'tracking agent performance metrics', 'MonthYear', 'date', 'month and year of performance record'],
    ['AgentPerformance', 'tracking agent performance metrics', 'TotalInteractions', 'numeric', 'number of interactions handled'],
    ['AgentPerformance', 'tracking agent performance metrics', 'AverageHandleTime', 'numeric', 'average interaction duration'],
    ['AgentPerformance', 'tracking agent performance metrics', 'CustomerSatisfactionAverage', 'numeric', 'average satisfaction score'],
    ['AgentPerformance', 'tracking agent performance metrics', 'ResolutionRate', 'numeric', 'percentage of issues resolved'],
    ['ServiceLevelAgreements', 'SLA tracking for customer types', 'SLAID', 'numeric', 'unique identifier for SLA'],
    ['ServiceLevelAgreements', 'SLA tracking for customer types', 'CustomerType', 'varchar', 'type of customer (standard, premium, enterprise)'],
    ['ServiceLevelAgreements', 'SLA tracking for customer types', 'MaxResponseTime', 'numeric', 'maximum allowed response time in minutes'],
    ['ServiceLevelAgreements', 'SLA tracking for customer types', 'MaxResolutionTime', 'numeric', 'maximum allowed resolution time in hours'],
    ['ServiceLevelAgreements', 'SLA tracking for customer types', 'EscalationProcess', 'text', 'details of escalation procedure']
]

# Create the DataFrame
df = pd.DataFrame(data, columns=['TableName', 'TableDesc', 'ColumnName', 'DataType', 'Description'])

# Save to CSV
csv_path = "assets/table_definition.csv"
df.to_csv(csv_path, index=False)

