'use client';

import { AnimatePresence, motion } from 'motion/react';
import { useSessionContext } from '@livekit/components-react';
import type { AppConfig } from '@/app-config';
import { DuetSessionView } from '@/components/app/duet-session-view';
import { WelcomeView } from '@/components/app/welcome-view';

const MotionWelcomeView = motion.create(WelcomeView);
const MotionSessionView = motion.create(DuetSessionView);

const VIEW_MOTION_PROPS = {
  variants: {
    visible: {
      opacity: 1,
    },
    hidden: {
      opacity: 0,
    },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
  transition: {
    duration: 0.5,
    ease: 'linear',
  },
};

interface ViewControllerProps {
  appConfig: AppConfig;
  topic: string;
  onStartCall: (topic: string) => void;
}

export function ViewController({ appConfig, topic, onStartCall }: ViewControllerProps) {
  const { isConnected } = useSessionContext();

  return (
    <AnimatePresence mode="wait">
      {/* Welcome view：单 trio 模式（良子×峰哥×老铁） */}
      {!isConnected && (
        <MotionWelcomeView
          key="welcome"
          {...VIEW_MOTION_PROPS}
          startButtonText={appConfig.startButtonText}
          onStartCall={onStartCall}
        />
      )}
      {/* Trio session view */}
      {isConnected && (
        <MotionSessionView
          key="session-trio"
          {...VIEW_MOTION_PROPS}
          interactive
          topic={topic}
          className="fixed inset-0"
        />
      )}
    </AnimatePresence>
  );
}
