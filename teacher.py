import cfg
import tensorflow as tf
import sys
import network
import dataManager

class teacher():
	def __init__(self,nTry):

		self.nTry = nTry
		self.outPut = network.outPut

		self.initLR = cfg.initLR
		self.global_step = tf.Variable(0, trainable=False)
		self.learningRate =tf.train.exponential_decay(self.initLR, self.global_step, 4000, 0.5)
		tf.summary.scalar('lR', self.learningRate)

		self.nEpoch = cfg.nEpoch

		self.y_outPut = tf.placeholder(tf.float32, [None, 10], name='y_outPut')
		self.usedSecondPoint = tf.placeholder(tf.float32, [None, 1], name='usedSecondPoint')
		self.costFunction()
		self.total_loss = tf.losses.get_total_loss()
		tf.summary.scalar('total_loss', self.total_loss)
		self.valLose = 0

		self.optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(self.total_loss, global_step=self.global_step)
		self.saver = tf.train.Saver()


		self.dataSet = dataManager.dataManager()
		self.dataSet.loadDataSet()
		#self.dataSet.validatDataSet()

		self.sess = tf.Session()
		self.merged = tf.summary.merge_all()
		self.train_writer = tf.summary.FileWriter('logs/expDay1/' + str(nTry) + "/train", self.sess.graph, flush_secs=10)
		self.test_writer = tf.summary.FileWriter('logs/expDay1/' + str(nTry) + "/test", self.sess.graph, flush_secs=10)
		self.sess.run(tf.global_variables_initializer())

	def costFunction(self):
		with tf.variable_scope("lose"):

			pointsDelta = self.y_outPut - self.outPut

			pointsLose =tf.reduce_mean(tf.reduce_sum(tf.square(pointsDelta), axis = 1))

			rightSlop = tf.div((self.outPut[:, 9]-self.outPut[:, 3]),(self.outPut[:, 8]-self.outPut[:, 2])+0.00001)
			leftSlop = tf.div((self.outPut[:, 5]-self.outPut[:, 9]),(self.outPut[:, 4]-self.outPut[:, 8])+0.00001)

			rightSlopL = tf.div((self.y_outPut[:, 9] - self.y_outPut[:, 3]), (self.y_outPut[:, 8] - self.y_outPut[:, 2])+0.00001)
			leftSlopL = tf.div((self.y_outPut[:, 5] - self.y_outPut[:, 9]), (self.y_outPut[:, 4] - self.y_outPut[:, 8])+0.00001)

			rightSlopPenalty = tf.reduce_mean(tf.abs(rightSlop - rightSlopL)) * 0.40
			leftSlopPenalty = tf.reduce_mean(tf.abs(leftSlop - leftSlopL)) * 0.40
			# diffSlop = tf.square(rightSlop-leftSloe)
			#
			# nZero = tf.reduce_sum(tf.ones_like(self.usedSecondPoint)-self.usedSecondPoint)
			# sumOfnotUsed =tf.reduce_sum(tf.multiply(tf.ones_like(self.usedSecondPoint)-self.usedSecondPoint, diffSlop))
			# penalty = sumOfnotUsed / nZero

			tf.losses.add_loss(pointsLose)
			tf.losses.add_loss(leftSlopPenalty)
			tf.losses.add_loss(rightSlopPenalty)
			tf.summary.scalar('pointsLose', pointsLose)
			tf.summary.scalar('leftSlopPenalty', leftSlopPenalty)
			tf.summary.scalar('rightSlopPenalty', rightSlopPenalty)

	def train(self):
		with tf.name_scope("train"):
			counter = 0
			valtrainSet, valOutPut ,used= self.dataSet.getValSet()
			for i in range(self.nEpoch):
				for j in range(self.dataSet.nBatch):
					trainSet, outPut, trainSetUsedScond = self.dataSet.getBatch()
					summary, opt = self.sess.run([self.merged, self.optimizer], feed_dict={network.nn.input: trainSet, self.y_outPut: outPut, self.usedSecondPoint: trainSetUsedScond})
					self.train_writer.add_summary(summary, counter)
					summary ,self.valLose = self.sess.run([self.merged, self.total_loss], feed_dict={network.nn.input: valtrainSet, self.y_outPut: valOutPut,self.usedSecondPoint: used})
					self.test_writer.add_summary(summary, counter)
					counter = counter + 1
				print("valLose--------------------------------------------",i,self.valLose)
			self.saver.save(self.sess, './models/day1/'+str(self.nTry))


train = teacher(sys.argv[1])

train.train()

