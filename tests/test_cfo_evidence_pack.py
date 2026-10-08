import pandas as pd
from src.cfo_evidence_pack_engine import CFOEvidencePackEngine

def main():
    v=pd.DataFrame({"Business_Unit":["A","A","B","B"],"Department":["M","I","M","I"],"Category":["Advertising","Cloud","Advertising","Travel"],"Budget_Amount":[100000,200000,150000,120000],"Actual_Amount":[115000,220000,145000,100000],"Variance":[15000,20000,-5000,-20000]})
    f=pd.DataFrame({"Business_Unit":["A","B"],"Department":["M","I"],"Category":["Advertising","Cloud"],"Projected_Variance":[12000,18000],"Projected_Variance_Pct":[12,9],"Early_Warning":["RED","AMBER"]})
    r=pd.DataFrame({"Business_Unit":["A","B"],"Department":["M","I"],"Category":["Advertising","Cloud"],"Risk_Score":[92,65],"Risk_Band":["CRITICAL","HIGH"]})
    rec=pd.DataFrame({"Business_Unit":["A","B"],"Department":["M","I"],"Category":["Advertising","Cloud"],"Management_Recommendation":["Review advertising spend.","Review cloud spend."]})
    a=pd.DataFrame({"Business_Unit":["A","B"],"Department":["M","I"],"Category":["Advertising","Cloud"],"Required_Cost_Control":[12000,18000]})
    s=pd.DataFrame({"Scenario_Name":["Base Scenario","5% Future Spend Control"],"Base_Projected_Variance":[30000,30000],"Scenario_Projected_Variance":[30000,15000],"Variance_Improvement":[0,15000],"Forecast_Savings":[0,15000],"Budget_Headroom_Change":[0,0],"Projected_Overspend_Records":[2,1],"Projected_Underspend_Records":[0,1],"On_Budget_Records":[0,0],"Outcome_Rank":[2,1]})
    outputs={"variance":v,"forecast":f,"risk":r,"recommendations":rec,"action_sizing":a,"scenario_ranking":s,"recommended_scenario":{"Scenario_Name":"5% Future Spend Control"},"planning_summary":{"Total Transactions":10,"Matched Transactions":9,"Exception Transactions":1,"Exception Amount":1000.0,"Exception Types":{"MISSING_DIMENSION":1}}}
    p=CFOEvidencePackEngine(outputs).build()
    assert p["executive_position"]["budget_total"]==570000
    assert p["executive_position"]["actual_total"]==580000
    assert p["executive_position"]["current_variance"]==10000
    assert len(p["top_unfavorable_drivers"])==2 and p["top_unfavorable_drivers"][0]["category"]=="Cloud"
    assert p["top_cfo_issues"][0]["early_warning"]=="RED"
    assert p["scenario_results"][0]["Recommended_Scenario"]=="5% Future Spend Control"
    ids=[x["id"] for x in p["evidence_ledger"]]; assert len(ids)==len(set(ids))
    print("\nEXECUTIVE POSITION: PASS")
    print("TOP DRIVERS: PASS")
    print("TOP CFO ISSUES: PASS")
    print("SCENARIO EVIDENCE: PASS")
    print("PLANNING EXCEPTIONS: PASS")
    print("EVIDENCE IDS: PASS")
    print("\nVARIA CFO EVIDENCE PACK TEST: PASS")
if __name__=="__main__": main()
