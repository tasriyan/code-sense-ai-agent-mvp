from pathlib import Path


def to_file(text: str, file_name: str):
    output_file = "D:\\src\\ai-agents\\code-navigator-agent\\books\\results\\llm_logs\\" + file_name
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write('\n\n')
        f.write("*" * 50)
        f.write(text)

def get_raw_claude_response():
    return """ 
Let me analyze the files using the provided tool.

```typescript
// Checking LoyaltyPointHistoryDTO.cs
let historyDto = get_code_by_filepath("D:\\src\\learning\\dotnet\\event-driven-course\\module5\\src\\PlantBasedPizza.LoyaltyPoints\\application\\PlantBasedPizza.LoyaltyPoints.Shared\\Core\\LoyaltyPointHistoryDTO.cs");

// Checking LoyaltyPointsDTO.cs
let pointsDto = get_code_by_filepath("D:\\src\\learning\\dotnet\\event-driven-course\\module5\\src\\PlantBasedPizza.LoyaltyPoints\\application\\PlantBasedPizza.LoyaltyPoints.Shared\\Core\\LoyaltyPointsDTO.cs");

// Checking repository interfacea
let repo = get_code_by_filepath("D:\\src\\learning\\dotnet\\event-driven-course\\module5\\src\\PlantBasedPizza.LoyaltyPoints\\application\\PlantBasedPizza.LoyaltyPoints.Shared\\Core\\ICustomerLoyaltyPointsRepository.cs");
```

{
"analysis_performed": [
    "Examined LoyaltyPointHistoryDTO.cs to understand transaction history structure",
    "Examined LoyaltyPointsDTO.cs to understand points data model",
    "Examined ICustomerLoyaltyPointsRepository.cs for data access patterns",
    "Identified DTO-based microservice architecture with repository pattern"
],
"suggested_service": "PlantBasedPizza.LoyaltyPoints.Shared",
"suggested_files": [
    {
        "action": "modify",
        "file_path": "/Core/LoyaltyPointsDTO.cs",
        "purpose": "Add expiration date field to points tracking"
    },
    {
        "action": "modify",
        "file_path": "/Core/LoyaltyPointHistoryDTO.cs",
        "purpose": "Add expiration date to historical records"
    },
    {
        "action": "create",
        "file_path": "/Core/PointsExpirationRule.cs",
        "purpose": "Define expiration calculation logic"
    }
],
"implementation_steps": [
    "Step 1: Add ExpirationDate property to LoyaltyPointsDTO",
    "Step 2: Add ExpirationDate property to LoyaltyPointHistoryDTO",
    "Step 3: Create PointsExpirationRule class to encapsulate expiration logic",
    "Step 4: Modify repository interface to support expiration date queries",
    "Step 5: Update existing repository implementations",
    "Step 6: Add expiration date calculation to points award logic"
],
"business_rationale": "Adding expiration dates to loyalty points encourages regular customer engagement and helps manage point liability. The implementation follows existing DTO patterns while adding necessary tracking for point expiration.",
"integration_points": [
    "LoyaltyPoints Service: Needs to calculate and store expiration dates",
    "Order Service: Must be notified of soon-to-expire points",
    "Customer Notification Service: Should alert customers of expiring points"
],
"code_examples": [
    {
        "file": "LoyaltyPointsDTO.cs",
        "code": "public class LoyaltyPointsDTO{public string CustomerId { get; set; }public int Points { get; set; } public DateTime? ExpirationDate { get; set; }public List<LoyaltyPointHistoryDTO> History { get; set; }}"
    },
    {
        "file": "PointsExpirationRule.cs",
        "code": "public class PointsExpirationRule { public DateTime CalculateExpirationDate(DateTime awardDate) { return awardDate.AddMonths(12); // Points expire after 1 year }}"
    }
],
"confidence_score": 0.90
}     
"""

def get_raw_claude__no_toolsresponse():
    return """ 
{
    "analysis_performed": [
        "Examined LoyaltyPointHistoryDTO.cs to understand point transaction structure",
        "Examined LoyaltyPointsDTO.cs to understand current points data model",
        "Examined ICustomerLoyaltyPointsRepository.cs for data access patterns",
        "Identified DTO and Repository patterns in use",
        "Noted existing timestamp tracking capabilities"
    ],
    "suggested_service": "PlantBasedPizza.LoyaltyPoints.Shared",
    "suggested_files": [
        {
            "action": "modify",
            "file_path": "PlantBasedPizza.LoyaltyPoints.Shared/Core/LoyaltyPointHistoryDTO.cs",
            "purpose": "Add expiration date field to point history records"
        },
        {
            "action": "modify",
            "file_path": "PlantBasedPizza.LoyaltyPoints.Shared/Core/CustomerLoyaltyPoints.cs",
            "purpose": "Add expiration tracking to domain entity"
        },
        {
            "action": "create",
            "file_path": "PlantBasedPizza.LoyaltyPoints.Shared/Core/PointsExpirationPolicy.cs",
            "purpose": "Define points expiration rules"
        }
    ],
    "implementation_steps": [
        "Step 1: Add ExpirationDate property to LoyaltyPointHistoryDto",
        "Step 2: Create PointsExpirationPolicy class to encapsulate expiration rules",
        "Step 3: Modify CustomerLoyaltyPoints entity to track active vs expired points",
        "Step 4: Update LoyaltyPointsDto constructor to include expiration information",
        "Step 5: Add expiration calculation logic in points award workflow",
        "Step 6: Add repository method to clean up expired points"
    ],
    "business_rationale": "Adding expiration dates to loyalty points encourages regular customer engagement and helps manage point liability. The implementation leverages existing history tracking while adding necessary expiration logic.",
    "integration_points": [
        "OrderService: Points awarded need expiration date calculation",
        "CustomerService: Must show valid vs expired points balance",
        "RedemptionService: Must verify points haven't expired before redemption"
    ],
    "code_examples": [
        {
            "file": "LoyaltyPointHistoryDTO.cs",
            "code": "public class LoyaltyPointHistoryDto{public DateTime DateTime { get; set; } public DateTime ExpirationDate { get; set; }public string OrderIdentifier { get; set; }public decimal OrderValue { get; set; }public decimal PointsAdded { get; set; }public bool IsExpired => DateTime.UtcNow > ExpirationDate;}"
        },
        {
            "file": "PointsExpirationPolicy.cs",
            "code": "public class PointsExpirationPolicy{public DateTime CalculateExpirationDate(DateTime awardDate){return awardDate.AddMonths(12); // Points expire after 1 year}}"
        }
    ],
    "confidence_score": 0.90
}  
"""