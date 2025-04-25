class Portfolio:
    def __init__(self, asset, fiat):
        self.asset = asset          # 可用资产
        self.fiat = fiat           # 可用法币
        self.next_day_available_asset = 0  # 下一个交易日可用的资产

    def valorisation(self, price):
        """计算当前投资组合的总价值"""
        return self.asset * price + self.fiat + self.next_day_available_asset * price

    def position(self, price):
        """计算当前仓位比例（基于可用资产）"""
        total = self.valorisation(price)
        return (self.asset * price) / total if total != 0 else 0

    def trade_to_position(self, position, price, trading_fees):
        """调整到目标仓位（0-1之间），考虑交易手续费"""
        if position < 0 or position > 1:
            raise ValueError("Position must be between 0 and 1")

        current_value = self.valorisation(price)
        if current_value <= 0:
            return  # 无资金时不操作

        # 计算目标资产数量
        target_asset = (position * current_value) / price
        asset_trade = target_asset - self.asset

        if asset_trade > 0:  # 买入操作
            # 计算需要的法币（考虑手续费）
            required_fiat = (asset_trade * price) / (1 - trading_fees)
            if required_fiat > self.fiat:  # 资金不足时按最大可买量调整
                asset_trade = (self.fiat * (1 - trading_fees)) / price
                required_fiat = self.fiat
            
            self.fiat -= required_fiat
            self.next_day_available_asset += asset_trade  # 次日到账

        elif asset_trade < 0:  # 卖出操作
            sell_amount = -asset_trade
            if sell_amount > self.asset:  # 超出持有量时按最大卖出
                sell_amount = self.asset
            
            self.asset -= sell_amount
            self.fiat += sell_amount * price * (1 - trading_fees)  # 立即到账

    def update_day(self):
        """每日结算，更新可用资产"""
        self.asset += self.next_day_available_asset
        self.next_day_available_asset = 0

    def get_portfolio_distribution(self):
        """获取资产分布信息"""
        return {
            "asset": self.asset,
            "fiat": self.fiat,
            "next_day_available_asset": self.next_day_available_asset
        }

    def __str__(self):
        return f"Portfolio(asset={self.asset}, fiat={self.fiat}, next_day_available_asset={self.next_day_available_asset})"

    def describe(self, price):
        print(f"Value: {self.valorisation(price)}, Position: {self.position(price)}")


class TargetPortfolio(Portfolio):
    """直接创建目标仓位的投资组合"""
    def __init__(self, position, value, price):
        if not 0 <= position <= 1:
            raise ValueError("Position must be between 0 and 1")
        super().__init__(
            asset=(position * value) / price,
            fiat=(1 - position) * value
        )