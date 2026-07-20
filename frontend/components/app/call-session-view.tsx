'use client';

import { useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import {
  useAgent,
  useLocalParticipant,
  useSessionContext,
  useSessionMessages,
} from '@livekit/components-react';
import {
  ChatTextIcon,
  MicrophoneIcon,
  MicrophoneSlashIcon,
  PhoneDisconnectIcon,
} from '@phosphor-icons/react';
import { AgentAudioVisualizerAura } from '@/components/agents-ui/agent-audio-visualizer-aura';
import { AgentChatTranscript } from '@/components/agents-ui/agent-chat-transcript';
import { LiangziAvatar } from '@/components/app/liangzi-avatar';
import { cn } from '@/lib/shadcn/utils';

const STATUS_TEXT: Record<string, string> = {
  connecting: '呼叫中…',
  initializing: '良子上线中…',
  listening: '在听你说…',
  thinking: '想词儿呢…',
  speaking: '说话中…',
  failed: '呼叫失败',
  disconnected: '通话结束',
};

interface CallSessionViewProps {
  supportsChatInput?: boolean;
  audioVisualizerColor?: `#${string}`;
  audioVisualizerColorShift?: number;
}

/**
 * 「给良子打电话」通话页：星云作底、头像居中、电话式圆钮控制栏。
 */
export function CallSessionView({
  supportsChatInput = true,
  audioVisualizerColor,
  audioVisualizerColorShift,
  ref,
  className,
}: React.ComponentProps<'section'> & CallSessionViewProps) {
  const session = useSessionContext();
  const { messages } = useSessionMessages(session);
  const { state: agentState, microphoneTrack } = useAgent();
  const { localParticipant } = useLocalParticipant();
  const [chatOpen, setChatOpen] = useState(false);

  const micEnabled = localParticipant.isMicrophoneEnabled;
  const statusText = STATUS_TEXT[agentState] ?? '通话中…';

  const toggleMic = () => {
    void localParticipant.setMicrophoneEnabled(!micEnabled);
  };

  return (
    <section
      ref={ref}
      className={cn('relative h-full w-full overflow-hidden bg-neutral-950', className)}
    >
      {/* 中部：星云 + 头像 + 名字/状态（整体上移，避开底部按钮） */}
      <div className="absolute inset-0 z-10 grid place-items-center pb-28">
        <div className="flex flex-col items-center">
          <div className="relative grid place-items-center">
            <AgentAudioVisualizerAura
              size="xl"
              state={agentState}
              themeMode="dark"
              color={audioVisualizerColor}
              colorShift={audioVisualizerColorShift}
              audioTrack={microphoneTrack}
              className="opacity-90"
            />
            <LiangziAvatar className="absolute size-28 text-5xl md:size-32 md:text-6xl" />
          </div>
          <p className="mt-6 text-2xl font-semibold text-white">良子</p>
          <p className="mt-2 flex items-center gap-2 text-sm text-neutral-400">
            {(agentState === 'connecting' || agentState === 'initializing') && (
              <span className="inline-block size-2 animate-pulse rounded-full bg-emerald-400" />
            )}
            {statusText}
          </p>
        </div>
      </div>

      {/* 聊天转写面板 */}
      <AnimatePresence>
        {chatOpen && (
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 24 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
            className="absolute inset-x-4 top-28 bottom-36 z-30 mx-auto max-w-xl overflow-hidden rounded-3xl border border-white/10 bg-black/70 backdrop-blur-md"
          >
            <AgentChatTranscript
              agentState={agentState}
              messages={messages}
              className="h-full w-full p-4 [&_.is-user>div]:rounded-2xl"
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* 底部：电话式控制钮 */}
      <div className="absolute inset-x-0 bottom-0 z-20 flex items-center justify-center gap-6 pb-12">
        <div className="flex flex-col items-center gap-2">
          <button
            type="button"
            onClick={toggleMic}
            aria-label={micEnabled ? '静音' : '取消静音'}
            className={cn(
              'grid size-14 place-items-center rounded-full transition-colors',
              micEnabled
                ? 'bg-white/10 text-white hover:bg-white/20'
                : 'bg-white text-neutral-900 hover:bg-neutral-200'
            )}
          >
            {micEnabled ? <MicrophoneIcon size={24} /> : <MicrophoneSlashIcon size={24} />}
          </button>
          <span className="text-xs text-neutral-500">{micEnabled ? '静音' : '已静音'}</span>
        </div>

        <div className="flex flex-col items-center gap-2">
          <button
            type="button"
            onClick={() => session.end()}
            aria-label="挂断"
            className="grid size-16 place-items-center rounded-full bg-red-500 text-white transition-colors hover:bg-red-400"
          >
            <PhoneDisconnectIcon size={28} />
          </button>
          <span className="text-xs text-neutral-500">挂断</span>
        </div>

        {supportsChatInput && (
          <div className="flex flex-col items-center gap-2">
            <button
              type="button"
              onClick={() => setChatOpen((v) => !v)}
              aria-label="聊天记录"
              className={cn(
                'grid size-14 place-items-center rounded-full transition-colors',
                chatOpen
                  ? 'bg-white text-neutral-900 hover:bg-neutral-200'
                  : 'bg-white/10 text-white hover:bg-white/20'
              )}
            >
              <ChatTextIcon size={24} />
            </button>
            <span className="text-xs text-neutral-500">记录</span>
          </div>
        )}
      </div>
    </section>
  );
}
