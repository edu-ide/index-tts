#!/usr/bin/env python3
"""
학습 모니터링 및 Slack 알림 스크립트
- TensorBoard 로그를 실시간으로 모니터링
- Loss spike, OOM 에러, 학습 완료 등 자동 알림
- WandB와 통합하여 중요 이벤트 추적

사용법:
    # 환경 변수 설정
    export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

    # 실행
    python monitor_training.py --log-dir /path/to/logs --check-interval 60
"""

import argparse
import os
import sys
import time
import requests
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

# TensorBoard 로그 파싱
try:
    from tensorboard.backend.event_processing import event_accumulator
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    print("[Error] TensorBoard not available. Install with: pip install tensorboard")
    sys.exit(1)

# WandB (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor training and send Slack alerts")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/mnt/sda1/models/index-tts-ko/checkpoints/logs"),
        help="TensorBoard log directory"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--loss-spike-threshold",
        type=float,
        default=1.5,
        help="Loss spike threshold (e.g., 1.5 = 50% increase)"
    )
    parser.add_argument(
        "--slack-webhook",
        type=str,
        default=os.environ.get("SLACK_WEBHOOK_URL", ""),
        help="Slack webhook URL (or set SLACK_WEBHOOK_URL env var)"
    )
    parser.add_argument(
        "--no-slack",
        action="store_true",
        help="Disable Slack notifications (only print to console)"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="indextts-korean",
        help="WandB project for logging alerts"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging"
    )
    return parser.parse_args()


class TrainingMonitor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.log_dir = args.log_dir
        self.slack_enabled = bool(args.slack_webhook) and not args.no_slack
        self.wandb_enabled = WANDB_AVAILABLE and not args.no_wandb

        self.last_checked_step = -1
        self.previous_loss = None
        self.loss_history: List[float] = []
        self.alert_history: List[Dict] = []

        # WandB 초기화
        if self.wandb_enabled:
            wandb.init(
                project=args.wandb_project,
                name=f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                job_type="monitoring",
            )
            print("[Info] WandB monitoring initialized")

    def get_latest_run(self) -> Optional[Path]:
        """최신 TensorBoard run 디렉토리 찾기"""
        if not self.log_dir.exists():
            return None

        runs = sorted(self.log_dir.glob("run_*"))
        return runs[-1] if runs else None

    def parse_tensorboard_logs(self, run_dir: Path) -> Optional[Dict]:
        """TensorBoard 로그 파싱"""
        try:
            ea = event_accumulator.EventAccumulator(str(run_dir))
            ea.Reload()

            # 사용 가능한 scalar tags 확인
            scalar_tags = ea.Tags().get('scalars', [])

            # Train loss 확인
            loss_tag = None
            if 'train/mel_loss' in scalar_tags:
                loss_tag = 'train/mel_loss'
            elif 'train/loss' in scalar_tags:
                loss_tag = 'train/loss'
            else:
                return None

            events = ea.Scalars(loss_tag)
            if not events:
                return None

            latest_event = events[-1]

            # Learning rate
            lr = None
            if 'train/lr' in scalar_tags:
                lr_events = ea.Scalars('train/lr')
                if lr_events:
                    lr = lr_events[-1].value

            return {
                "step": latest_event.step,
                "loss": latest_event.value,
                "lr": lr,
                "timestamp": latest_event.wall_time,
            }

        except Exception as e:
            print(f"[Error] Failed to parse TensorBoard logs: {e}")
            return None

    def send_slack_alert(self, message: str, level: str = "info"):
        """Slack 알림 전송"""
        if not self.slack_enabled:
            return

        # 이모지 선택
        emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "🚨",
            "success": "✅",
        }.get(level, "📢")

        payload = {
            "text": f"{emoji} {message}",
            "username": "IndexTTS Training Monitor",
            "icon_emoji": ":robot_face:",
        }

        try:
            response = requests.post(
                self.args.slack_webhook,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            print(f"[Slack] Sent: {message}")
        except requests.exceptions.RequestException as e:
            print(f"[Error] Failed to send Slack message: {e}")

    def check_loss_spike(self, current_loss: float) -> bool:
        """Loss spike 감지"""
        if self.previous_loss is None:
            return False

        # Loss가 threshold 이상 증가했는지 확인
        if current_loss > self.previous_loss * self.args.loss_spike_threshold:
            return True

        return False

    def check_for_errors(self) -> Optional[str]:
        """학습 에러 확인 (로그 파일 검사)"""
        # TODO: 실제 학습 로그 파일 경로로 교체
        log_file = Path("/tmp/training.log")
        if not log_file.exists():
            return None

        # 최근 100줄만 확인
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()[-100:]

            # OOM 에러 검사
            for line in lines:
                if "CUDA out of memory" in line or "OOM" in line:
                    return "CUDA OOM (Out of Memory) detected!"

                if "RuntimeError" in line or "Exception" in line:
                    return f"Runtime error detected: {line.strip()}"

        except Exception as e:
            print(f"[Error] Failed to read log file: {e}")

        return None

    def log_alert(self, alert_type: str, message: str, metrics: Optional[Dict] = None):
        """알림 기록 (콘솔 + WandB)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert = {
            "timestamp": timestamp,
            "type": alert_type,
            "message": message,
            "metrics": metrics or {},
        }
        self.alert_history.append(alert)

        # 콘솔 출력
        print(f"\n[{timestamp}] [{alert_type.upper()}] {message}")
        if metrics:
            for key, value in metrics.items():
                print(f"  - {key}: {value}")

        # WandB 로깅
        if self.wandb_enabled:
            wandb.log({
                f"alert/{alert_type}": 1,
                **{f"alert/{k}": v for k, v in (metrics or {}).items()}
            })

    def monitor_step(self):
        """1회 모니터링 실행"""
        # 최신 run 찾기
        latest_run = self.get_latest_run()
        if not latest_run:
            print("[Info] No TensorBoard runs found yet")
            return

        # 로그 파싱
        data = self.parse_tensorboard_logs(latest_run)
        if not data:
            return

        # 새로운 step인지 확인
        if data["step"] == self.last_checked_step:
            return

        self.last_checked_step = data["step"]
        current_loss = data["loss"]
        self.loss_history.append(current_loss)

        # Loss spike 감지
        if self.check_loss_spike(current_loss):
            message = (
                f"Loss spike detected at step {data['step']}!\n"
                f"Previous: {self.previous_loss:.4f} → Current: {current_loss:.4f}\n"
                f"Increase: {(current_loss / self.previous_loss - 1) * 100:.1f}%"
            )
            self.send_slack_alert(message, level="warning")
            self.log_alert("loss_spike", message, {
                "step": data["step"],
                "previous_loss": self.previous_loss,
                "current_loss": current_loss,
            })

        # 에러 확인
        error_msg = self.check_for_errors()
        if error_msg:
            self.send_slack_alert(f"Training error detected!\n{error_msg}", level="error")
            self.log_alert("error", error_msg, {"step": data["step"]})

        # 진행상황 출력
        lr_str = f"LR: {data['lr']:.2e}" if data['lr'] else ""
        print(f"[Monitor] Step {data['step']:,}: Loss {current_loss:.4f} {lr_str}")

        self.previous_loss = current_loss

    def run(self):
        """모니터링 루프 실행"""
        print(f"[Info] Starting training monitor")
        print(f"[Info] Log directory: {self.log_dir}")
        print(f"[Info] Check interval: {self.args.check_interval}s")
        print(f"[Info] Loss spike threshold: {self.args.loss_spike_threshold}x")
        print(f"[Info] Slack alerts: {'Enabled' if self.slack_enabled else 'Disabled'}")
        print(f"[Info] Press Ctrl+C to stop\n")

        # 시작 알림
        if self.slack_enabled:
            self.send_slack_alert(
                f"Training monitoring started\nLog dir: {self.log_dir}",
                level="info"
            )

        try:
            while True:
                self.monitor_step()
                time.sleep(self.args.check_interval)

        except KeyboardInterrupt:
            print("\n[Info] Monitoring stopped by user")

            # 종료 알림
            if self.slack_enabled and self.loss_history:
                final_loss = self.loss_history[-1]
                self.send_slack_alert(
                    f"Training monitoring stopped\n"
                    f"Final loss: {final_loss:.4f}\n"
                    f"Total steps monitored: {self.last_checked_step}",
                    level="info"
                )

            if self.wandb_enabled:
                wandb.finish()

        except Exception as e:
            error_msg = f"Monitoring crashed: {str(e)}"
            print(f"[Error] {error_msg}")

            if self.slack_enabled:
                self.send_slack_alert(error_msg, level="error")

            if self.wandb_enabled:
                wandb.finish()

            raise


def main():
    args = parse_args()

    # Slack webhook 확인
    if not args.no_slack and not args.slack_webhook:
        print("[Warning] Slack webhook not configured. Set SLACK_WEBHOOK_URL env var or use --slack-webhook")
        print("[Warning] Running without Slack notifications")
        args.no_slack = True

    # 모니터 시작
    monitor = TrainingMonitor(args)
    monitor.run()


if __name__ == "__main__":
    main()
