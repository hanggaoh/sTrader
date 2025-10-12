import pytest
from data.sentiment import analyze_sentiment


def test_chinese_sentiment_analysis():
    """
    Tests the Chinese sentiment analysis model with specific examples.
    """
    # 1. Test with a strongly positive headline
    positive_title = "公司利润大幅增长，远超市场预期"
    positive_content = "本季度财报显示，公司净利润同比增长了80%，各项业务指标均创历史新高。"
    positive_score = analyze_sentiment(positive_title, positive_content)
    print(f"Positive test case score: {positive_score}")
    assert positive_score > 0.5  # Expect a strongly positive score

    # 2. Test with a strongly negative headline
    negative_title = "公司季度亏损严重，面临退市风险"
    negative_content = "由于市场需求疲软和成本上升，公司本季度出现巨额亏损，现金流紧张，警告可能触发退市条件。"
    negative_score = analyze_sentiment(negative_title, negative_content)
    print(f"Negative test case score: {negative_score}")
    assert negative_score < -0.5  # Expect a strongly negative score

    # 3. Test with a neutral headline
    neutral_title = "中国人民银行今日进行100亿元逆回购操作"
    neutral_content = "为维护银行体系流动性合理充裕，中国人民银行今日以利率招标方式开展了100亿元逆回购操作。"
    neutral_score = analyze_sentiment(neutral_title, neutral_content)
    print(f"Neutral test case score: {neutral_score}")
    assert -0.1 < neutral_score < 0.1  # Expect a score very close to zero
